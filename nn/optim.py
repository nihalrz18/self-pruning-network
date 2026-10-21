"""Mask-aware optimizers.

The subtle part of masked training is NOT the forward pass; it is the optimizer state.

Naive pipeline:  W_eff = W * M  ->  grad(W) = M * grad(W_eff)  ->  Adam(W).
What goes wrong with an ordinary optimizer:
  1. Momentum leak: a weight pruned at step t still has m_t != 0. With g = 0 afterwards,
     Adam keeps applying  lr * m_hat / (sqrt(v_hat)+eps)  and keeps *moving the stored
     weight* (its moments decay only geometrically). The mask hides it in the forward
     pass, but the stored value is no longer a "real zero".
  2. Stale-state revival: a revived connection resumes with its OLD m, v and a global
     step counter. Its first updates are driven by gradient statistics from a different
     phase of training (under different weights) and by a bias-correction factor
     (1 - beta^t) that assumes t steps of history this connection does not have.

Our rules (identical for SGD-momentum and Adam):
  * Gradient is multiplied by the mask (defence in depth; the `masked` op already does).
  * Moments are updated ONLY on active entries and are forced to exactly 0 on masked
    ones; stored weights at masked entries are exactly 0.0 after every step.
  * Adam keeps a PER-ELEMENT step counter t_ij used in the bias correction; it is reset
    to 0 whenever an entry is pruned or revived (`on_mask_change`). A revived
    connection therefore behaves like a brand-new parameter: its first bias-corrected
    Adam step has magnitude ~ lr, exactly as in step 1 of a fresh run.
"""
from __future__ import annotations

import numpy as np

from engine import Tensor


class _Optimizer:
    def __init__(self, params: list[Tensor]):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def _state_arrays(self, p: Tensor) -> list[np.ndarray]:
        raise NotImplementedError

    def on_mask_change(self, p: Tensor, changed: np.ndarray):
        """Call after the mask of `p` changed. `changed` is a bool array of entries that
        were pruned OR revived; their optimizer state is wiped to a pristine zero."""
        for s in self._state_arrays(p):
            s[changed] = 0.0


class SGD(_Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.mom, self.wd = lr, momentum, weight_decay
        self.buf = {id(p): np.zeros_like(p.data) for p in self.params}

    def _state_arrays(self, p):
        return [self.buf[id(p)]]

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            active = None
            if p.mask is not None:
                active = p.mask > 0
                g = g * p.mask
            if self.wd:
                g = g + self.wd * p.data
            buf = self.buf[id(p)]
            buf *= self.mom
            buf += g
            if active is not None:
                buf *= active  # momentum of a pruned entry is exactly 0
                p.data = np.where(active, p.data - self.lr * buf, 0.0)
            else:
                p.data = p.data - self.lr * buf


class Adam(_Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
        self.t = {id(p): np.zeros_like(p.data) for p in self.params}  # per-element steps

    def _state_arrays(self, p):
        return [self.m[id(p)], self.v[id(p)], self.t[id(p)]]

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            i = id(p)
            m, v, t = self.m[i], self.v[i], self.t[i]
            active = np.ones(p.shape, bool) if p.mask is None else p.mask > 0
            g = p.grad * active
            t += active  # in-place: only active entries advance their clock
            m[:] = np.where(active, self.b1 * m + (1 - self.b1) * g, 0.0)
            v[:] = np.where(active, self.b2 * v + (1 - self.b2) * g * g, 0.0)
            tt = np.maximum(t, 1.0)
            m_hat = m / (1 - self.b1 ** tt)
            v_hat = v / (1 - self.b2 ** tt)
            new = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.wd:
                new = new - self.lr * self.wd * p.data  # decoupled (AdamW)
            p.data = np.where(active, new, 0.0)
