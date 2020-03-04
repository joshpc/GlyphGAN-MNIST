import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision
from torchvision import datasets, transforms

def get_mnist_dataloaders(batch_size=128, character_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
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

    # Use SubsetRandomSampler, given a set of indices, to be the dataloader.
    character_indices = {}
    for i in range(len(train_data)):
        label = train_data[i][1]

        if label in character_indices:
            character_indices[label].append(i)
        else:
            character_indices[label] = [i]

    train_loaders = []

    for value in character_classes:
        data_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(character_indices[value]))
        train_loaders.append(data_loader)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    all_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loaders, test_loader, all_loader
