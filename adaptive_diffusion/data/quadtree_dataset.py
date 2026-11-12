"""
Dataset de PyTorch: cada muestra es un quadtree binario serializado como
tokens (nivel, y, x, estado), más la etiqueta del dígito.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from data.binary_mnist import load_binary_mnist
from structures.binary_quadtree import build_levels, to_tokens


class QuadtreeMNIST(Dataset):
    def __init__(self, train=True, limit=None):
        imgs, labels = load_binary_mnist(train=train)
        if limit:
            imgs, labels = imgs[:limit], labels[:limit]
        levels = build_levels(imgs)
        self.depth = len(levels) - 1
        self.tokens = [torch.from_numpy(to_tokens([lv[i] for lv in levels])) for i in range(len(imgs))]
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]
