import torch
import torch.nn as nn                 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model.ViT import ViT, HopfieldViT
from model import *
# Define the Vision Transformer model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='PyTorch ViT Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")

parser.add_argument('--net', default='vit')
# parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='256', type=int)
parser.add_argument('--size', default="32", type=int)
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dim', default="512", type=int)
parser.add_argument('--depth', default="6", type=int)
parser.add_argument('--heads', default="8", type=int)
parser.add_argument('--mlp_dim', default='512', type=int, help="parameter for convmixer")
parser.add_argument('--data', default='cifar10')
parser.add_argument('--dropout', default='0.1', type=float)
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()

if args.wandb:
    import wandb

    wandb.init(project="vit")
    wandb.config.update(args)



# Dataset
if args.data == 'cifar10':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
    train_dataset = torchvision.datasets.CIFAR10(root='~/data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='~/data', train=False, transform=transform, download=True)
    img_size = 32
    in_channels = 3
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
else:
    raise Exception('data not found')




# model
if args.net == 'vit':
    model = ViT(
        image_size = img_size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        channels = in_channels,
        dropout = args.dropout
    )
elif args.net == 'vit_hopfield':
    model = HopfieldViT(
        image_size = img_size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        channels = in_channels,
        dropout = args.dropout
    )


model = model.to(device)

# data parallel
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

scalar = torch.cuda.amp.GradScaler()

for epoch in range(args.n_epochs):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        pbar.set_description(f'Loss: {loss.item()}')
    scheduler.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if args.wandb:
        wandb.log({'epoch': epoch, 'loss': loss.item(), 'accuracy': 100 * correct / total, 'lr': scheduler.get_last_lr()[0]})
    print(f'Accuracy: {100 * correct / total}')


