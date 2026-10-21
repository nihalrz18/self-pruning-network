"""Part 3: one self-pruning run (train + prune simultaneously, budget hit exactly).

    python -m train.run_part3 [--sparsity 0.9] [--criterion fisher] [--regrow 0.0] [--seed 0]

Writes results/part3_run.json and results/part3_run.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .experiment import run_one, DEFAULTS
from .run_part2 import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", default="fisher")
    ap.add_argument("--schedule", default="cubic")
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    dense = run_one(None, 0.0, a.seed)
    print(f"dense baseline : test acc {dense['test_acc']:.4f}")
    res, model, hist, pruner, data = run_one(a.criterion, a.sparsity, a.seed, schedule=a.schedule,
                                             regrow_frac=a.regrow, verbose=True, return_model=True)
    print(f"\n{a.criterion} @ target {a.sparsity:.0%}: achieved sparsity {res['achieved_sparsity']:.4f}, "
          f"test acc {res['test_acc']:.4f}  (cost vs dense: {res['test_acc'] - dense['test_acc']:+.4f})")
    for i, l in enumerate(model.linears()):
        print(f"  layer {i}: {int(l.mask.sum())}/{l.mask.size} live weights "
              f"({1 - l.mask.mean():.1%} pruned); stored zeros at masked slots: "
              f"{bool(np.all(l.weight.data[l.mask == 0] == 0.0))}")
    json.dump(dict(result=res, dense=dense, history=hist,
                   events=pruner.log if pruner else []), open(os.path.join(RESULTS, "part3_run.json"), "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = hist["epoch"]
    ax[0].plot(ep, hist["sparsity"], color="C3")
    ax[0].axvspan(DEFAULTS["prune_start_ep"], DEFAULTS["prune_end_ep"], alpha=0.12, label="pruning window")
    ax[0].set(xlabel="epoch", ylabel="sparsity", title=f"{a.schedule} sparsity ramp (target {a.sparsity:.0%})")
    ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["test_acc"], label="test")
    ax[1].axhline(dense["test_acc"], ls="--", color="gray", label="dense test")
    ax[1].set(xlabel="epoch", ylabel="accuracy", ylim=(0.8, 1.01), title="Accuracy while pruning")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "part3_run.png"), dpi=130)
    print("wrote results/part3_run.json, results/part3_run.png")


if __name__ == "__main__":
    main()
