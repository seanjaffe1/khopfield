import torch
import torch.nn as nn                 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import *
# Define the Vision Transformer model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

import pandas as pd


    
# Define the Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim, dim, num_heads, img_size, patch_size, in_channels=3):
        super(VisionTransformer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.hopfield = KHopfield(N=dim, n=embed_dim * self.num_patches)
        self.fc = nn.Linear(embed_dim * self.num_patches, num_classes)
        self.num_heads = num_heads

    def forward(self, x):
        x1 = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x2 = x1.permute(0, 2, 3, 1)  # (batch_size, num_patches_h, num_patches_w, embed_dim)
        x3 = x2.reshape(x2.size(0), -1, x2.size(-1))  # (batch_size, num_patches, embed_dim)
        
        x4 = x3 + self.positional_embedding  # Add positional embedding
        # combine second and third dimension
        x5 = x4.flatten(1, 2)
        x6 = self.hopfield(x5, self.num_heads)
        x7 = x6.mean(dim=2)  # Global average pooling
        x8 = self.fc(x7)
        return x8
    
    def to(self, device):
        super(VisionTransformer, self).to(device)
        self.hopfield = self.hopfield.to(device)
        return self
    
# Uses #num_heads k=1 hopfield networks, rather than k=num_heads-hopfield networks
class VisionTransformerV(nn.Module):
    def __init__(self, num_classes, embed_dim, dim, num_heads, img_size, patch_size, in_channels=3):
        super(VisionTransformerV, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.hopfields = nn.ModuleList([KHopfield(N=dim, n=embed_dim * self.num_patches) for _ in range(num_heads)])
        self.fc = nn.Linear(embed_dim * self.num_patches, num_classes)
        self.num_heads = num_heads

    def forward(self, x):
        x1 = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x2 = x1.permute(0, 2, 3, 1)  # (batch_size, num_patches_h, num_patches_w, embed_dim)
        x3 = x2.reshape(x2.size(0), -1, x2.size(-1))  # (batch_size, num_patches, embed_dim)
        
        x4 = x3 + self.positional_embedding  # Add positional embedding
        # combine second and third dimension
        x5 = x4.flatten(1, 2)
        x6 = [self.hopfields[i](x5, 1) for i in range(self.num_heads)]
        # take average of all heads
        x6 = torch.stack(x6, dim=2).squeeze()
        x7 = x6.mean(dim=2)  # Global average pooling
        x8 = self.fc(x7)
        return x8
    
def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def get_model_and_data(
        data = 'mnist',
        model = 'hopfield',
        batch_size = 256,
        heads = 4,
        dim=256,
        embed_dim=1024):
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(root='~/data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='~/data', train=False, transform=transforms.ToTensor(), download=True)
        img_size = 28
        in_channels = 1
        num_classes = 10
        patch_size = 7

    elif data  == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root='~/data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='~/data', train=False, transform=transform, download=True)
        img_size = 32
        in_channels = 3
        num_classes = 10
        patch_size = 16
    else:
        raise Exception('data not found')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    if model == 'hopfield':
        model = VisionTransformer(
            num_classes = num_classes, 
            embed_dim = embed_dim, 
            dim = dim,
            num_heads = heads, 
            img_size = img_size, 
            patch_size = patch_size, 
            in_channels = in_channels,
        )
    elif model == 'hopfieldV':
        model = VisionTransformerV(
            num_classes = num_classes, 
            embed_dim = embed_dim, 
            dim = dim,
            num_heads = heads, 
            img_size = img_size, 
            patch_size = patch_size, 
            in_channels = in_channels,
        )
    elif model == 'vit':
        model = SimpleViT(
            image_size = img_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = 1,
            heads = heads,
            mlp_dim = 1024
        )
    return model, train_loader, test_loader


def run_experiment(train_loader, test_loader, model, num_epochs=30, k=1):
    # Initialize the model and optimizer
    model = model.to(device)
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    val_accuracy = []
    # Training loop
    for epoch in range(num_epochs):

        # show loss in tqdm
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'K: {k}, Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}')
            break
        # validate
        accuracy = validate(model, test_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}')
        val_accuracy.append(accuracy)

    print('Training finished!')

    return val_accuracy


ks = [1, 4, 8, 16]

df = pd.DataFrame(columns = ['k', 'accuracy', 'model', 'epoch'])
datas = [ 'cifar10', 'mnist']
model_names = ['hopfieldV', 'hopfield', 'vit']

datas = ['mnist']
model_names = ['vit']
for data in datas:
    for model_name in model_names:
        for k in ks:
            try:
                model, train_loader, test_loader = get_model_and_data(data = data, model = model_name, batch_size = 256, heads = k, dim=100)
                val_accuracy  = run_experiment(train_loader, test_loader, model, num_epochs=30, k =k )
                # add every val accuracy to dataframe
                for i, acc in enumerate(val_accuracy):
                    df = df.append({'k': k, 'accuracy': acc, 'model': 'hopfield', 'epoch': i}, ignore_index=True)
                
                # save dataframe
                df.to_csv(f'./results/{data}_{model_name}_heads.csv')
            except Exception as e:
                print(e)
                print(f'failed on {data}, {model_name}, {k}')
                continue