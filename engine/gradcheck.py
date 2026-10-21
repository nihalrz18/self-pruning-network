"""Finite-difference gradient checking."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences of scalar-valued f(ndarray) at x."""
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check_grad(fn, *arrays, eps: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7,
               seed: int = 0):
    """Compare autodiff vs numerical gradients of  sum(fn(*tensors) * R)  for a fixed
    random projection R (so non-scalar outputs exercise every output element with a
    different weight, which catches transposed / mis-broadcast gradients).

    Returns max relative error across all inputs; raises AssertionError on mismatch.
    """
    rng = np.random.RandomState(seed)
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    probe = fn(*[Tensor(a) for a in arrays])
    R = rng.randn(*probe.shape) if probe.shape else np.array(1.0)

    def scalar(*xs):
        return float((fn(*[Tensor(x) for x in xs]).data * R).sum())

    ts = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*ts)
    out.backward(R if out.shape else np.array(1.0))

    worst = 0.0
    for k, t in enumerate(ts):
        def f_k(x, k=k):
            xs = list(arrays)
            xs[k] = x
            return scalar(*xs)

        num = numerical_grad(f_k, arrays[k], eps)
        ana = t.grad if t.grad is not None else np.zeros_like(arrays[k])
        err = np.abs(ana - num)
        tol = atol + rtol * np.maximum(np.abs(ana), np.abs(num))
        rel = (err / (np.abs(ana) + np.abs(num) + 1e-12)).max()
        worst = max(worst, float(rel))
        if not np.all(err <= tol):
            raise AssertionError(f"grad mismatch for input {k}: max abs err {err.max():.3e}\n"
                                 f"analytic=\n{ana}\nnumeric=\n{num}")
    return worst
