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
        data = 'cifar10',
        model_name = 'vit_hopfield',
        batch_size = 256,
        heads = 4):
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='~/data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='~/data', train=False, transform=transforms.ToTensor(), download=True)
        img_size = 28
        in_channels = 1
        num_classes = 10

    elif data  == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])

        train_dataset = torchvision.datasets.CIFAR10(root='~/data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='~/data', train=False, transform=transform, download=True)
        img_size = 32
        in_channels = 3
        num_classes = 10
    else:
        raise Exception('data not found')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    patch_size= 4
    if model_name == 'vit':
        model = ViT(
            image_size = img_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 128,
            depth = 2,
            heads = heads,
            mlp_dim = 512,
            channels = in_channels,
        )
    elif model_name == 'vit_hopfield':
        model = HopfieldViT(
            image_size = img_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 128,
            depth = 2,
            heads = heads,
            mlp_dim = 512,
            channels = in_channels,
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
        # validate
        accuracy = validate(model, test_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}')
        val_accuracy.append(accuracy)

    print('Training finished!')

    return val_accuracy


ks = [1, 4, 8, 16]

df = pd.DataFrame(columns = ['k', 'accuracy', 'model', 'epoch'])
datas = ['mnist'] #[ 'cifar10']
model_names = ['vit', 'vit_hopfield']

for data in datas:
    for model_name in model_names:
        for k in ks:
            try:
                model, train_loader, test_loader = get_model_and_data(data = data, model_name = model_name, batch_size = 256, heads = k)
                val_accuracy  = run_experiment(train_loader, test_loader, model, num_epochs=100, k =k )
                # add every val accuracy to dataframe
                for i, acc in enumerate(val_accuracy):
                    df = df.append({'k': k, 'accuracy': acc, 'model': 'hopfield', 'epoch': i}, ignore_index=True)
                
                # save dataframe
                df.to_csv(f'./results/{data}_{model_name}_heads.csv')
            except Exception as e:
                print(e)
                print(f'failed on {data}, {model_name}, {k}')
                continue