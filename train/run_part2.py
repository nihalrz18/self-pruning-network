"""Part 2: train a dense MLP with our Adam on our autodiff engine.

    python -m train.run_part2 [--dataset digits] [--epochs 60] [--seed 0]

Produces results/part2_learning_curve.png, results/part2_curve.json and
results/part2_init_study.json (the weight-initialisation justification).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from engine import Tensor
from nn import build_mlp
from .data import load_dataset
from .trainer import fit

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HIDDEN = [256, 128]


def activation_stats(model, x):
    """Std of each Linear layer's pre-activation on a batch (forward signal health)."""
    model(Tensor(x))
    return [float(l.out.data.std()) for l in model.linears()]


def init_study(data, seed=0, epochs=15):
    """He init vs. too-small / too-large / zero weights.

    Claim to demonstrate: with Var(W)=2/fan_in the pre-activation scale stays O(1) through
    depth, so Adam trains from step 0; other scales vanish/explode the signal."""
    rows = []
    for name, scale in [("zeros", 0.0), ("0.1 x He", 0.1), ("He (ours)", 1.0), ("10 x He", 10.0)]:
        m = build_mlp([data.n_features, *HIDDEN, data.n_classes], seed=seed)
        for l in m.linears():
            l.weight.data = l.weight.data * scale
        stats = activation_stats(m, data.x_train[:256])
        try:
            h = fit(m, data, epochs=epochs, seed=seed)
            final = dict(final_train_loss=h["train_loss"][-1], final_test_acc=h["test_acc"][-1], nan=False)
        except FloatingPointError:
            final = dict(final_train_loss=None, final_test_acc=None, nan=True)
        rows.append(dict(init=name, preact_std_per_layer=stats, **final))
        print(f"  init={name:10s} preact std/layer={np.round(stats, 3)}  {final}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--skip-init-study", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    data = load_dataset(a.dataset)
    print(f"dataset={a.dataset} train={len(data.x_train)} val={len(data.x_val)} test={len(data.x_test)}")
    model = build_mlp([data.n_features, *HIDDEN, data.n_classes], seed=a.seed)
    n_w = sum(l.weight.size for l in model.linears())
    print(f"MLP {data.n_features}-{'-'.join(map(str, HIDDEN))}-{data.n_classes}: {n_w} weights; "
          f"He init, Adam lr={a.lr}")
    hist = fit(model, data, epochs=a.epochs, lr=a.lr, seed=a.seed, verbose=True)
    assert all(np.isfinite(hist["train_loss"])), "NaN/inf in training"
    print(f"final: train acc {hist['train_acc'][-1]:.4f}  val acc {hist['val_acc'][-1]:.4f}  "
          f"test acc {hist['test_acc'][-1]:.4f}   (no NaNs; max grad-norm {max(hist['grad_norm']):.2f})")

    json.dump(hist, open(os.path.join(RESULTS, "part2_curve.json"), "w"), indent=1)
    if not a.skip_init_study:
        print("Initialisation study (15 epochs each):")
        json.dump(init_study(data, a.seed), open(os.path.join(RESULTS, "part2_init_study.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = hist["epoch"]
    ax[0].plot(ep, hist["train_loss"], label="train loss")
    ax[0].plot(ep, hist["test_loss"], label="test loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss (dense MLP, Adam)")
    ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].plot(ep, hist["test_acc"], label="test")
    ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy", ylim=(0.8, 1.01))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "part2_learning_curve.png"), dpi=130)
    print("wrote results/part2_learning_curve.png")


if __name__ == "__main__":
    main()
