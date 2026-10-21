"""Datasets. scikit-learn is used ONLY here, ONLY to load the bundled 8x8 digits images
(`sklearn.datasets.load_digits`, no download needed). No sklearn model, optimizer,
metric or gradient is used anywhere in this repository."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int

    @property
    def n_features(self):
        return self.x_train.shape[1]


def _split(x, y, n_classes, data_seed, frac=(0.6, 0.15, 0.25)) -> Split:
    rng = np.random.RandomState(data_seed)
    idx = rng.permutation(len(x))
    n_tr, n_va = int(frac[0] * len(x)), int(frac[1] * len(x))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8  # standardise with TRAIN statistics only
    f = lambda a: (a - mu) / sd
    return Split(f(x[tr]), y[tr], f(x[va]), y[va], f(x[te]), y[te], n_classes)


def load_digits_split(data_seed: int = 0) -> Split:
    from sklearn.datasets import load_digits  # data loading only

    d = load_digits()
    return _split(d.data.astype(np.float64), d.target.astype(np.int64), 10, data_seed)


def make_spirals(n_per_class: int = 300, n_classes: int = 3, noise: float = 0.2,
                 data_seed: int = 0) -> Split:
    rng = np.random.RandomState(data_seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        th = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + rng.randn(n_per_class) * noise
        xs.append(np.c_[r * np.sin(th), r * np.cos(th)])
        ys.append(np.full(n_per_class, c))
    return _split(np.vstack(xs), np.concatenate(ys).astype(np.int64), n_classes, data_seed)


def load_dataset(name: str, data_seed: int = 0) -> Split:
    return {"digits": load_digits_split, "spirals": make_spirals}[name](data_seed=data_seed)
