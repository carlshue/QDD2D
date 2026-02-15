"""
File: mnist_loader.py
Project: Adaptive Hierarchical Diffusion

Description:
    Provides utilities to download, preprocess and load MNIST
    dataset samples using PyTorch torchvision.

Responsibilities:
    - Download MNIST if not present
    - Apply normalization
    - Provide train/test DataLoaders
"""
"""
    🔹 data/mnist_loader.py

Debe contener:

def get_mnist_dataloaders(batch_size: int):
    ...


Incluye:

Descarga automática

Transform → ToTensor()

Normalización opcional

Devuelve train_loader, test_loader

Nada más.
    
"""


"""
File: mnist_loader.py
Project: Adaptive Hierarchical Diffusion

Description:
    Provides utilities to download, preprocess and load MNIST dataset samples.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
