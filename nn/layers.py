"""Composable layers built on the Part 1 engine."""
from __future__ import annotations

import numpy as np

from engine import Tensor, matmul, masked
from .init import he_normal, xavier_normal


class Module:
    def parameters(self) -> list[Tensor]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """y = x @ (W * M) + b   with a connectivity mask M (all ones = dense).

    The mask is also stored on `self.weight.mask` so a mask-aware optimizer can see it.
    After `backward()` the layer exposes what the pruner needs:
      * `last_input`   (B, in)  the activations fed in,
      * `w_eff.grad`   (in, out) the DENSE gradient dL/dW_eff (non-zero at pruned slots),
      * `out.grad`     (B, out)  per-sample output deltas (scaled by 1/B for a mean loss).
    """

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.RandomState,
                 init: str = "he"):
        w0 = he_normal(rng, fan_in, fan_out) if init == "he" else xavier_normal(rng, fan_in, fan_out)
        self.weight = Tensor(w0, requires_grad=True)
        self.bias = Tensor(np.zeros(fan_out), requires_grad=True)
        self.weight.mask = np.ones_like(self.weight.data)
        self.fan_in, self.fan_out = fan_in, fan_out
        self.last_input = None
        self.w_eff: Tensor | None = None
        self.out: Tensor | None = None

    @property
    def mask(self) -> np.ndarray:
        return self.weight.mask

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        self.last_input = x.data
        self.w_eff = masked(self.weight, self.weight.mask)
        self.out = matmul(x, self.w_eff) + self.bias
        return self.out

    def set_mask(self, new_mask: np.ndarray):
        """Install a mask and make pruned weights *genuine* zeros in storage."""
        self.weight.mask = new_mask.astype(np.float64)
        self.weight.data = self.weight.data * self.weight.mask


class _Act(Module):
    fn = None

    def forward(self, x):
        return getattr(x, self.fn)()


class ReLU(_Act):
    fn = "relu"


class Tanh(_Act):
    fn = "tanh"


class Sigmoid(_Act):
    fn = "sigmoid"


class GELU(_Act):
    fn = "gelu"


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        for l in self.layers:
            x = l(x)
        return x

    def linears(self) -> list[Linear]:
        return [l for l in self.layers if isinstance(l, Linear)]


def build_mlp(sizes: list[int], seed: int = 0, activation: str = "relu") -> Sequential:
    """MLP with `len(sizes)-1` Linear layers and an activation between them (none after
    the last: it emits logits)."""
    rng = np.random.RandomState(seed)
    act = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "gelu": GELU}[activation]
    init = "he" if activation in ("relu", "gelu") else "xavier"
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(a, b, rng, init))
        if i < len(sizes) - 2:
            layers.append(act())
    return Sequential(*layers)
