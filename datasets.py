import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

def get_mnist_dataloaders(batch_size=128):
    """
    MNIST dataloader with (32, 32) sized images.
    """
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('./data', train=True, download=True, transform=all_transforms)
    test_data = datasets.MNIST('./data', train=False, transform=all_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader