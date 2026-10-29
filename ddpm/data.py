"""MNIST / Fashion-MNIST loading, scaled to [-1, 1]. Data is downloaded, never committed."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}


def load(name: str = "mnist", root: str = "data", train: bool = True, subset: int | None = None):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])  # [0,1] -> [-1,1]
    ds = DATASETS[name](root, train=train, download=True, transform=tfm)
    if subset:
        ds = Subset(ds, torch.arange(min(subset, len(ds))).tolist())
    return ds


def loader(ds, batch_size: int, shuffle: bool = True, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=shuffle, num_workers=0)
