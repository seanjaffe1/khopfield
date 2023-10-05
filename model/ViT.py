import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .lml3 import LML3
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)


        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class KhopfieldAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        qvk_dim = dim_head  #
        inner_dim = dim_head *  heads 
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = lambda x: self.k_softmax(x, k=heads)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, qvk_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    @staticmethod
    def ssm(x, k, beta):
        # x in b x n
        return LML3(N=k, n_iter = 200,eps = 1e-3)(beta * x)


    def k_softmax(self, beta, x, k):
        # x is b x N
        # beta is scalar
        # k is int
        # returns b x N x k
        assert(k>=1)
        result = torch.zeros(x.shape[0], x.shape[1], k).to(x.device)
        result[:, :, 0] =self.ssm(x, 1, beta)
        last_ssm = result[:, :, 0]
        for i in range(1, k):
            new_ssm  = self.ssm(x, i+1, beta)
            result[:, :, i] = new_ssm - last_ssm
            last_ssm = new_ssm.clone()
        return result

    def forward(self, x):
        x = self.norm(x)


        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n d -> b n d'), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.squeeze()

        dots = rearrange(dots, 'b n d -> (b n) d')

        attn = self.k_softmax(beta = 1, x=dots, k=self.heads)


        attn = rearrange(attn, '(b n) d h -> b h n d', b = x.shape[0], n = x.shape[1])
        
        

        attn = self.dropout(attn)
        
        
        v = rearrange(v, 'b n d -> b 1 n d ')

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:

            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class KHopfieldTransformer(Transformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__(dim, depth, heads, dim_head, mlp_dim, dropout = dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                KhopfieldAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
    
class HopfieldViT(ViT):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        self.transformer = KHopfieldTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)