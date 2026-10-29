"""Sample-quality metrics built on a small MNIST classifier trained in this repo.

NOT the standard FID: FID uses Inception-v3 features and ImageNet statistics, which are not
meaningful for 28x28 grayscale digits. `fid_proxy` is the same Frechet distance formula applied
to the 64-d penultimate features of our own classifier, so numbers are only comparable within
this repo (same classifier checkpoint), never against published FID values.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch import nn


class DigitClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
        self.fc1, self.fc2 = nn.Linear(32 * 7 * 7, 64), nn.Linear(64, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def train_classifier(train_ds, test_ds, epochs: int = 2, seed: int = 0) -> tuple[DigitClassifier, float]:
    from .data import loader

    torch.manual_seed(seed)
    clf = DigitClassifier()
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in loader(train_ds, 128, seed=seed):
            opt.zero_grad()
            F.cross_entropy(clf(x), y).backward()
            opt.step()
    return clf.eval(), accuracy(clf, test_ds)


@torch.no_grad()
def accuracy(clf: DigitClassifier, ds) -> float:
    from .data import loader

    correct = n = 0
    for x, y in loader(ds, 512, shuffle=False):
        correct += (clf(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


def frechet_distance(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


@torch.no_grad()
def feature_stats(clf: DigitClassifier, images: torch.Tensor):
    f = clf.features(images).numpy().astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


@torch.no_grad()
def fid_proxy(clf: DigitClassifier, a: torch.Tensor, b: torch.Tensor) -> float:
    return frechet_distance(*feature_stats(clf, a), *feature_stats(clf, b))


@torch.no_grad()
def sample_report(clf: DigitClassifier, samples: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
    """Mean max-softmax confidence, predicted-class histogram, and (if conditional) label agreement."""
    probs = F.softmax(clf(samples), 1)
    pred = probs.argmax(1)
    out = {
        "mean_confidence": float(probs.max(1).values.mean()),
        "class_histogram": torch.bincount(pred, minlength=10).tolist(),
    }
    if labels is not None:
        out["label_agreement"] = float((pred == labels).float().mean())
    return out
