"""Sparse-aware inference path (pure NumPy) used for the Part 4 cost evidence.

`SparseLinear` stores only the live weights in CSC-by-output-column order
(col_ptr, row_idx, vals). Its forward gathers just the needed input features and
multiplies by the stored values, so the number of multiplies executed is exactly
batch * nnz -- not batch * in * out. A counter on the object records the multiplies that
actually run (the size of the product array), so FLOP numbers are measured, not computed
from the sparsity.

HONESTY NOTE: this is a *NumPy gather* kernel. Dense `x @ W` goes to a tuned BLAS and runs
at a large fraction of peak even when W is full of zeros, so a pure-NumPy sparse kernel does
fewer FLOPs but is often NOT faster in wall-clock until sparsity is very high. We report
both numbers (see experiments in train/run_part4.py and DESIGN.md).
`dense_masked_forward` is the dense-times-zero shortcut; it saves nothing and is included
only as the comparison.
"""
from __future__ import annotations

import numpy as np

from nn import Linear, Sequential


class SparseLinear:
    def __init__(self, weight: np.ndarray, mask: np.ndarray, bias: np.ndarray):
        w = weight * mask
        self.in_features, self.out_features = w.shape
        rows, cols = np.nonzero(mask > 0)  # np.nonzero is row-major; we want column-major
        order = np.lexsort((rows, cols))
        rows, cols = rows[order], cols[order]
        self.row_idx = rows.astype(np.intp)
        self.vals = w[rows, cols].copy()
        counts = np.bincount(cols, minlength=self.out_features)
        self.col_ptr = np.concatenate([[0], np.cumsum(counts)])
        self.nonempty = np.flatnonzero(counts)
        self.starts = self.col_ptr[:-1][self.nonempty]
        self.bias = bias.copy()
        self.nnz = int(len(self.vals))
        self.mults = 0  # measured multiplies executed so far

    def forward(self, x: np.ndarray) -> np.ndarray:
        prod = x[:, self.row_idx] * self.vals  # (B, nnz): exactly nnz multiplies per sample
        self.mults += prod.size
        out = np.zeros((x.shape[0], self.out_features))
        if self.nnz:
            out[:, self.nonempty] = np.add.reduceat(prod, self.starts, axis=1)
        return out + self.bias


_ACTS = {"ReLU": lambda z: np.maximum(z, 0.0), "Tanh": np.tanh,
         "Sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
         "GELU": lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))}


class SparseMLP:
    """Inference-only, autodiff-free sparse version of a trained (pruned) Sequential."""

    def __init__(self, model: Sequential):
        self.ops = []
        for layer in model.layers:
            if isinstance(layer, Linear):
                self.ops.append(SparseLinear(layer.weight.data, layer.mask, layer.bias.data))
            else:
                self.ops.append(_ACTS[type(layer).__name__])

    def forward(self, x: np.ndarray) -> np.ndarray:
        for op in self.ops:
            x = op.forward(x) if isinstance(op, SparseLinear) else op(x)
        return x

    @property
    def linears(self):
        return [o for o in self.ops if isinstance(o, SparseLinear)]

    def mults(self) -> int:
        return sum(l.mults for l in self.linears)

    def reset_counters(self):
        for l in self.linears:
            l.mults = 0

    def nnz(self) -> int:
        return sum(l.nnz for l in self.linears)


def dense_masked_forward(model: Sequential, x: np.ndarray) -> np.ndarray:
    """Dense-times-zero shortcut: same math, but multiplies every zero too. Does NOT count
    as a speed-up (it performs B*in*out multiplies per layer regardless of sparsity)."""
    for layer in model.layers:
        if isinstance(layer, Linear):
            x = x @ (layer.weight.data * layer.mask) + layer.bias.data
        else:
            x = _ACTS[type(layer).__name__](x)
    return x


def compact_dense_forward(model: Sequential):
    """Structured gain that unstructured pruning sometimes gives for free: a hidden unit
    with no live OUTGOING weight is never read; one with no live INCOMING weight emits the
    constant act(bias), which is folded into the next layer's bias. Both can be deleted,
    leaving smaller *dense* BLAS matmuls with identical outputs.

    Returns (forward_fn, kept_units_per_hidden_layer, dense_mults_per_sample)."""
    lin = model.linears()
    acts = [_ACTS[type(l).__name__] for l in model.layers if not isinstance(l, Linear)]
    ws = [(l.weight.data * l.mask).copy() for l in lin]
    bs = [l.bias.data.copy() for l in lin]
    kept = []
    for i in range(len(ws) - 1):
        has_in = np.abs(ws[i]).sum(0) > 0
        has_out = np.abs(ws[i + 1]).sum(1) > 0
        const = has_out & ~has_in  # constant act(b_j) feeding the next layer
        if const.any():
            bs[i + 1] += acts[i](bs[i][const]) @ ws[i + 1][const]
        kept.append(np.flatnonzero(has_in & has_out))
    cw, cb = [], []
    for i, (w, b) in enumerate(zip(ws, bs)):
        if i > 0:
            w = w[kept[i - 1]]
        if i < len(ws) - 1:
            w, b = w[:, kept[i]], b[kept[i]]
        cw.append(w)
        cb.append(b)

    def fwd(x):
        for i, (w, b) in enumerate(zip(cw, cb)):
            x = x @ w + b
            if i < len(cw) - 1:
                x = acts[i](x)
        return x

    return fwd, [len(k) for k in kept], sum(w.size for w in cw)
