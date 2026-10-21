"""Weight initialisers.

He / Kaiming (std = sqrt(2 / fan_in)) for ReLU-family nets: ReLU zeroes half of the
pre-activations, halving the second moment of the signal, so Var(W) = 2/fan_in keeps
E[h^2] constant layer to layer (neither explodes nor vanishes with depth).
Xavier / Glorot (std = sqrt(2 / (fan_in + fan_out))) for tanh / sigmoid, which are
roughly linear (gain 1) around 0 and want forward *and* backward variance preserved.
Weights are stored as (fan_in, fan_out) because the layer computes x @ W.
"""
import numpy as np


def he_normal(rng: np.random.RandomState, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def xavier_normal(rng: np.random.RandomState, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
