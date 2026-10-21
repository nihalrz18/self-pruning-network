"""Reverse-mode autodiff tensor.

A `Tensor` wraps a float64 ndarray, remembers the tensors it was computed from
(`_parents`) and a closure (`_backward`) that pushes `self.grad` into the parents'
grads. `backward()` runs those closures in reverse topological order.

Design notes
------------
* Every node keeps its `.grad` after `backward()` (not freed). The pruner relies on
  this: the gradient of the *effective* (masked) weight is the "what-if" gradient
  used for regrowth, and the gradient of a layer's output gives per-sample Fisher
  information. Cost: memory. See DESIGN.md (bottlenecks).
* Gradients accumulate (`+=`), so a tensor used twice in a graph, or a parameter
  across calls without `zero_grad()`, behaves like the mathematical sum of paths.
* Broadcasting: every binary op reduces the incoming gradient back to each
  operand's shape with `unbroadcast`.
"""
from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    (N, D) + (D,) : grad (N, D) -> sum over axis 0 -> (D,)
    (N, D) + (N, 1): grad (N, D) -> sum over axis 1, keepdims -> (N, 1)
    """
    if grad.shape == tuple(shape):
        return grad
    # leading axes that broadcasting prepended
    n_extra = grad.ndim - len(shape)
    if n_extra > 0:
        grad = grad.sum(axis=tuple(range(n_extra)))
    # axes that were size 1 in the operand and got stretched
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 1000  # make ndarray defer to Tensor's reflected operators

    def __init__(self, data, requires_grad: bool = False, _parents=(), _op: str = ""):
        self.data = np.array(data, dtype=np.float64)  # copy: Tensors own their storage
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._parents = tuple(_parents)
        self._op = _op
        self._backward = lambda: None
        # Optional connectivity mask (1.0 = alive, 0.0 = pruned). Set by MaskedLinear;
        # read by the optimizers. `None` means "dense parameter".
        self.mask: np.ndarray | None = None

    # ------------------------------------------------------------------ basics
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None

    def _accum(self, g: np.ndarray):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    # --------------------------------------------------------------- backward
    def backward(self, grad: np.ndarray | None = None):
        """Backpropagate from this tensor. Scalars get dL/dL = 1 by default."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() on a non-scalar needs an explicit grad")
            grad = np.ones_like(self.data)
        # Iterative DFS post-order -> topological order (no recursion-limit issues).
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad = np.zeros_like(self.data) if self.grad is None else self.grad
        self.grad = self.grad + np.asarray(grad, dtype=np.float64)
        for node in reversed(topo):
            node._backward()

    # --------------------------------------------------------------- operators
    def __add__(self, other):
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(as_tensor(other), self)

    def __mul__(self, other):
        return mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(as_tensor(other), self)

    def __neg__(self):
        return neg(self)

    def __pow__(self, p):
        return power(self, p)

    def __matmul__(self, other):
        return matmul(self, other)

    def sum(self, axis=None, keepdims=False):
        return sum_(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def relu(self):
        return relu(self)

    def tanh(self):
        return tanh(self)

    def sigmoid(self):
        return sigmoid(self)

    def gelu(self):
        return gelu(self)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def reshape(self, *shape):
        return reshape(self, shape)


def as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data, parents, op) -> Tensor:
    out = Tensor(data, requires_grad=any(p.requires_grad for p in parents),
                 _parents=parents, _op=op)
    return out


# ============================================================ elementwise binary
def add(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = _make(a.data + b.data, (a, b), "add")

    def _backward():
        if a.requires_grad:
            a._accum(unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b._accum(unbroadcast(out.grad, b.shape))

    out._backward = _backward
    return out


def sub(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = _make(a.data - b.data, (a, b), "sub")

    def _backward():
        if a.requires_grad:
            a._accum(unbroadcast(out.grad, a.shape))
        if b.requires_grad:
            b._accum(unbroadcast(-out.grad, b.shape))

    out._backward = _backward
    return out


def mul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = _make(a.data * b.data, (a, b), "mul")

    def _backward():
        if a.requires_grad:
            a._accum(unbroadcast(out.grad * b.data, a.shape))
        if b.requires_grad:
            b._accum(unbroadcast(out.grad * a.data, b.shape))

    out._backward = _backward
    return out


def div(a, b):
    a, b = as_tensor(a), as_tensor(b)
    out = _make(a.data / b.data, (a, b), "div")

    def _backward():
        if a.requires_grad:
            a._accum(unbroadcast(out.grad / b.data, a.shape))
        if b.requires_grad:
            b._accum(unbroadcast(-out.grad * a.data / (b.data ** 2), b.shape))

    out._backward = _backward
    return out


def neg(a):
    a = as_tensor(a)
    out = _make(-a.data, (a,), "neg")

    def _backward():
        if a.requires_grad:
            a._accum(-out.grad)

    out._backward = _backward
    return out


def power(a, p: float):
    """a ** p for a constant scalar exponent p."""
    a = as_tensor(a)
    out = _make(a.data ** p, (a,), f"pow{p}")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad * p * a.data ** (p - 1))

    out._backward = _backward
    return out


# ================================================================== matmul
def matmul(a, b):
    a, b = as_tensor(a), as_tensor(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul supports 2-D @ 2-D only (use reshape for vectors)")
    out = _make(a.data @ b.data, (a, b), "matmul")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad @ b.data.T)
        if b.requires_grad:
            b._accum(a.data.T @ out.grad)

    out._backward = _backward
    return out


# ================================================================ reductions
def _expand_reduced(g: np.ndarray, in_shape, axis, keepdims):
    """Reshape a reduced gradient so it broadcasts back over `in_shape`."""
    if axis is None:
        return g.reshape((1,) * len(in_shape)) if not keepdims else g
    if keepdims:
        return g
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(ax % len(in_shape) for ax in axes)
    shape = list(in_shape)
    for ax in axes:
        shape[ax] = 1
    return g.reshape(shape)


def sum_(a, axis=None, keepdims=False):
    a = as_tensor(a)
    out = _make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum")

    def _backward():
        if a.requires_grad:
            g = _expand_reduced(out.grad, a.shape, axis, keepdims)
            a._accum(np.broadcast_to(g, a.shape))

    out._backward = _backward
    return out


def mean(a, axis=None, keepdims=False):
    a = as_tensor(a)
    out = _make(a.data.mean(axis=axis, keepdims=keepdims), (a,), "mean")
    count = a.data.size // max(out.data.size, 1)

    def _backward():
        if a.requires_grad:
            g = _expand_reduced(out.grad, a.shape, axis, keepdims) / count
            a._accum(np.broadcast_to(g, a.shape))

    out._backward = _backward
    return out


def reshape(a, shape):
    a = as_tensor(a)
    out = _make(a.data.reshape(shape), (a,), "reshape")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad.reshape(a.shape))

    out._backward = _backward
    return out


# ============================================================== non-linearities
def relu(a):
    a = as_tensor(a)
    out = _make(np.maximum(a.data, 0.0), (a,), "relu")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad * (a.data > 0))  # subgradient 0 at x == 0

    out._backward = _backward
    return out


def tanh(a):
    a = as_tensor(a)
    t = np.tanh(a.data)
    out = _make(t, (a,), "tanh")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad * (1.0 - t * t))

    out._backward = _backward
    return out


def _stable_sigmoid(x):
    # exp only ever sees non-positive arguments -> no overflow
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid(a):
    a = as_tensor(a)
    s = _stable_sigmoid(a.data)
    out = _make(s, (a,), "sigmoid")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad * s * (1.0 - s))

    out._backward = _backward
    return out


_GELU_C = np.sqrt(2.0 / np.pi)


def gelu(a):
    """tanh approximation: 0.5 x (1 + tanh(c (x + 0.044715 x^3)))."""
    a = as_tensor(a)
    x = a.data
    u = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    out = _make(0.5 * x * (1.0 + t), (a,), "gelu")

    def _backward():
        if a.requires_grad:
            du = _GELU_C * (1.0 + 3 * 0.044715 * x ** 2)
            a._accum(out.grad * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du))

    out._backward = _backward
    return out


def exp(a):
    a = as_tensor(a)
    e = np.exp(a.data)
    out = _make(e, (a,), "exp")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad * e)

    out._backward = _backward
    return out


def log(a):
    a = as_tensor(a)
    out = _make(np.log(a.data), (a,), "log")

    def _backward():
        if a.requires_grad:
            a._accum(out.grad / a.data)

    out._backward = _backward
    return out


# ================================================================== masking
def masked(w, mask: np.ndarray):
    """Effective weight  W_eff = W * M   (M is a constant 0/1 array).

    Forward: a pruned entry contributes an exact 0.0, no matter what value is stored
    in W.
    Backward to the *parameter*: dL/dW = M * dL/dW_eff  -- the true chain-rule
    derivative. L does not depend on a masked entry, so its derivative is exactly 0.
    The *dense* "what-if" gradient dL/dW_eff (non-zero at pruned entries) is still
    available as `out.grad` and is used only for regrowth decisions.
    """
    w = as_tensor(w)
    m = np.asarray(mask, dtype=np.float64)
    out = _make(w.data * m, (w,), "masked")

    def _backward():
        if w.requires_grad:
            w._accum(out.grad * m)

    out._backward = _backward
    return out


# =================================================== softmax / cross-entropy
def softmax(a, axis=-1):
    """Numerically stable softmax (shift by the row max)."""
    a = as_tensor(a)
    z = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=axis, keepdims=True)
    out = _make(s, (a,), "softmax")

    def _backward():
        if a.requires_grad:
            g = out.grad
            # J^T g for softmax: s * (g - sum(g * s))
            a._accum(s * (g - (g * s).sum(axis=axis, keepdims=True)))

    out._backward = _backward
    return out


def log_softmax(a, axis=-1):
    a = as_tensor(a)
    z = a.data - a.data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=axis, keepdims=True))
    ls = z - lse
    out = _make(ls, (a,), "log_softmax")

    def _backward():
        if a.requires_grad:
            g = out.grad
            a._accum(g - np.exp(ls) * g.sum(axis=axis, keepdims=True))

    out._backward = _backward
    return out


def softmax_cross_entropy(logits, targets: np.ndarray, reduction: str = "mean"):
    """Fused, numerically stable  -log softmax(logits)[target].

    Using log-sum-exp avoids `log(softmax)` underflow to -inf, and the fused backward
    is the clean  (softmax - onehot) / N  instead of a chain through 1/p.
    """
    logits = as_tensor(logits)
    y = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - lse
    nll = -logp[np.arange(n), y]
    val = nll.mean() if reduction == "mean" else nll.sum()
    out = _make(val, (logits,), "softmax_xent")
    scale = 1.0 / n if reduction == "mean" else 1.0

    def _backward():
        if logits.requires_grad:
            g = np.exp(logp)
            g[np.arange(n), y] -= 1.0
            logits._accum(out.grad * g * scale)

    out._backward = _backward
    return out
