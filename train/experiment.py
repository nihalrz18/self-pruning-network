"""One configurable training run, shared by Part 3, Part 4 and the ablations."""
from __future__ import annotations

import numpy as np

from nn import build_mlp
from prune import SelfPruner
from .data import load_dataset
from .trainer import fit, global_sparsity

HIDDEN = [256, 128]
DEFAULTS = dict(dataset="digits", epochs=80, batch_size=64, lr=3e-3, prune_start_ep=10,
                prune_end_ep=50, every=10)


def run_one(criterion: str | None, sparsity: float, seed: int, *, schedule="cubic",
            regrow_frac=0.0, scope="layer", ema=0.9, reset_state=True, verbose=False, return_model=False, **over):
    """criterion=None -> dense baseline (no pruner). Returns a dict of final metrics."""
    cfg = {**DEFAULTS, **over}
    data = load_dataset(cfg["dataset"], data_seed=seed)  # split varies with seed (paired across methods)
    model = build_mlp([data.n_features, *HIDDEN, data.n_classes], seed=seed)
    spe = int(np.ceil(len(data.x_train) / cfg["batch_size"]))
    pruner = None
    if criterion is not None and sparsity > 0:
        pruner = SelfPruner(criterion, sparsity, schedule=schedule,
                            start_step=cfg["prune_start_ep"] * spe, end_step=cfg["prune_end_ep"] * spe,
                            every=cfg["every"], regrow_frac=regrow_frac, scope=scope, ema=ema,
                            reset_state=reset_state, seed=seed)
    hist = fit(model, data, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
               seed=seed, pruner=pruner, verbose=verbose,
               eval_every=1 if verbose else cfg["epochs"])
    res = dict(criterion=criterion or "dense", target_sparsity=sparsity, seed=seed,
               achieved_sparsity=global_sparsity(model), test_acc=hist["test_acc"][-1],
               val_acc=hist["val_acc"][-1], train_acc=hist["train_acc"][-1],
               test_loss=hist["test_loss"][-1], schedule=schedule, regrow_frac=regrow_frac, scope=scope,
               ema=ema, reset_state=reset_state)
    if pruner is not None and pruner.log:
        res["mean_churn"] = float(np.mean([l["churn"] for ev in pruner.log for l in ev["layers"]]))
    if return_model:
        return res, model, hist, pruner, data
    return res
