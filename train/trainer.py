"""Mini-batch training loop (own code; uses only the Part 1 engine + nn/ + prune/)."""
from __future__ import annotations

import numpy as np

import engine as E
from engine import Tensor
from nn import Adam, SGD, Sequential, build_mlp
from .data import Split


def predict_logits(model: Sequential, x: np.ndarray, chunk: int = 1024) -> np.ndarray:
    return np.vstack([model(Tensor(x[i:i + chunk])).data for i in range(0, len(x), chunk)])


def evaluate(model, x, y) -> tuple[float, float]:
    z = predict_logits(model, x)
    zs = z - z.max(1, keepdims=True)
    logp = zs - np.log(np.exp(zs).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean()), float((z.argmax(1) == y).mean())


def global_sparsity(model: Sequential) -> float:
    """Fraction of prunable (weight-matrix) entries that are masked out."""
    tot = sum(l.mask.size for l in model.linears())
    alive = sum(float(l.mask.sum()) for l in model.linears())
    return 1.0 - alive / tot


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0):
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(name)


def fit(model: Sequential, data: Split, epochs: int, batch_size: int = 64, lr: float = 3e-3,
        optimizer: str = "adam", seed: int = 0, pruner=None, weight_decay: float = 0.0,
        verbose: bool = False, log_every: int = 10, eval_every: int = 1) -> dict:
    """Train with mini-batches. If `pruner` is given it is told about every step:
       observe()  after backward (reads grads, updates importance statistics)
       update()   after the optimizer step (may change masks + wipe optimizer state)
    Test data is only *evaluated*, never used for any decision."""
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    if pruner is not None:
        pruner.attach(model, opt)
    rng = np.random.RandomState(seed + 10_000)  # batch order: separate stream from init
    n = len(data.x_train)
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_acc", "test_acc",
                            "test_loss", "sparsity", "grad_norm"]}
    step = 0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        gn = []
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = E.softmax_cross_entropy(model(Tensor(data.x_train[b])), data.y_train[b])
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            gn.append(np.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters())))
            if pruner is not None:
                pruner.observe(step)
            opt.step()
            if pruner is not None:
                pruner.update(step)
            step += 1
        if ep % eval_every and ep != epochs:
            continue
        trl, tra = evaluate(model, data.x_train, data.y_train)
        _, vaa = evaluate(model, data.x_val, data.y_val)
        tel, tea = evaluate(model, data.x_test, data.y_test)
        for k, v in zip(hist, [ep, trl, tra, vaa, tea, tel, global_sparsity(model), float(np.mean(gn))]):
            hist[k].append(v)
        if verbose and (ep % log_every == 0 or ep == 1 or ep == epochs):
            print(f"ep {ep:3d} loss {trl:.4f} train {tra:.3f} val {vaa:.3f} test {tea:.3f} "
                  f"sparsity {hist['sparsity'][-1]:.3f} |g| {hist['grad_norm'][-1]:.3f}")
    hist["steps_per_epoch"] = int(np.ceil(n / batch_size))
    return hist
