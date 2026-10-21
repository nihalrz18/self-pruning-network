"""Ablations + a direct test of the importance-criterion derivation (bonus evidence).

    python -m train.run_ablations [--sparsity 0.98] [--seeds 10]

Outputs: results/ablations.csv, ablations.md, saliency_vs_true_loss.json
"""
from . import parallel  # noqa: F401  (pins BLAS threads first)

import argparse
import json
import os

import numpy as np

from engine import Tensor
import engine as E
from nn import build_mlp
from prune.criteria import LayerStats, importance
from .experiment import run_one, HIDDEN
from .parallel import run_many
from .run_part2 import RESULTS
from .run_part4 import write_csv
from .stats import paired_stats
from .data import load_dataset
from .trainer import fit

REF = dict(criterion="fisher", schedule="cubic", regrow_frac=0.0, scope="layer", ema=0.9, reset_state=True)

VARIANTS = {  # name -> overrides of REF
    "reference (fisher, cubic, no regrow, per-layer, ema .9)": {},
    "criterion = magnitude": dict(criterion="magnitude"),
    "criterion = taylor |w g|": dict(criterion="taylor"),
    "criterion = taylor2 (|wg| + 1/2 w^2 F)": dict(criterion="taylor2"),
    "criterion = movement": dict(criterion="movement"),
    "schedule = linear": dict(schedule="linear"),
    "schedule = one-shot at end of window": dict(schedule="oneshot"),
    "regrowth 10%": dict(regrow_frac=0.1),
    "regrowth 30%": dict(regrow_frac=0.3),
    "regrowth 30%, NO optimizer-state reset (the bug)": dict(regrow_frac=0.3, reset_state=False),
    "scope = global ranking": dict(scope="global"),
    "magnitude + global ranking": dict(criterion="magnitude", scope="global"),
    "ema 0.5": dict(ema=0.5),
    "ema 0.99": dict(ema=0.99),
}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def saliency_vs_true_loss(seed=0, n_probe=400, epochs=25):
    """How well does each criterion rank the TRUE loss increase of deleting one weight?

    Train a dense net, accumulate importance statistics over the final epoch's batches, then
    for random live weights measure  true_dL = L(w_i := 0) - L(w)  on the training set and
    compute Spearman rank correlation with each criterion's score."""
    data = load_dataset("digits", data_seed=seed)
    model = build_mlp([data.n_features, *HIDDEN, data.n_classes], seed=seed)
    fit(model, data, epochs=epochs, seed=seed, eval_every=epochs)
    lins = model.linears()
    rng = np.random.RandomState(seed)
    stats = [LayerStats(l.weight.shape, {"taylor", "fisher"}, rng) for l in lins]
    # statistics from a pass of mini-batches at the trained point (weights fixed)
    perm = rng.permutation(len(data.x_train))
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        model.zero_grad()
        E.softmax_cross_entropy(model(Tensor(data.x_train[b])), data.y_train[b]).backward()
        for l, st in zip(lins, stats):
            st.observe(l, 0.9)
    def full_loss():
        return float(E.softmax_cross_entropy(model(Tensor(data.x_train)), data.y_train).data)

    base = full_loss()
    scores = {c: [] for c in ("magnitude", "taylor", "fisher", "taylor2")}
    truth = []
    for _ in range(n_probe):
        li = rng.randint(len(lins))
        l = lins[li]
        idx = (rng.randint(l.weight.shape[0]), rng.randint(l.weight.shape[1]))
        old = l.weight.data[idx]
        l.weight.data[idx] = 0.0
        truth.append(full_loss() - base)
        l.weight.data[idx] = old
        for c in scores:
            scores[c].append(importance(c, l, stats[li])[idx])
    out = {c: spearman(np.array(v), np.array(truth)) for c, v in scores.items()}
    out["n_probe"] = n_probe
    out["median_true_dL"] = float(np.median(truth))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.98)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    os.makedirs(RESULTS, exist_ok=True)

    jobs = []
    for name, ov in VARIANTS.items():
        for s in seeds:
            jobs.append(dict(sparsity=a.sparsity, seed=s, **{**REF, **ov}, tag=name))
    raw = run_many(jobs, desc="ablations")
    for r, j in zip(raw, jobs):
        r["variant"] = j["tag"]

    ref_name = next(iter(VARIANTS))
    ref = {r["seed"]: r["test_acc"] for r in raw if r["variant"] == ref_name}
    rows = []
    for name in VARIANTS:
        acc = np.array([r["test_acc"] for r in sorted((r for r in raw if r["variant"] == name), key=lambda r: r["seed"])])
        d = acc - np.array([ref[s] for s in seeds])
        st = paired_stats(d) if name != ref_name else dict(mean=0, ci_lo=0, ci_hi=0, p_perm=1, wins=0, losses=0, ties=len(seeds))
        churn = float(np.mean([r.get("mean_churn", 0.0) for r in raw if r["variant"] == name]))
        rows.append(dict(variant=name, sparsity=a.sparsity, test_acc_mean=acc.mean(), test_acc_std=acc.std(ddof=1),
                         diff_vs_ref_pp=100 * st["mean"], ci_lo_pp=100 * st["ci_lo"], ci_hi_pp=100 * st["ci_hi"],
                         p_perm=st["p_perm"], mean_event_churn=churn))
    write_csv(os.path.join(RESULTS, "ablations.csv"), rows)
    lines = [f"Ablations at {a.sparsity:.0%} sparsity, {len(seeds)} seeds (paired against the reference row).\n",
             "| variant | test acc | sd | diff vs ref (pp) | 95% CI (pp) | p | churn/event |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['test_acc_mean']:.4f} | {r['test_acc_std']:.4f} | "
                     f"{r['diff_vs_ref_pp']:+.2f} | [{r['ci_lo_pp']:+.2f}, {r['ci_hi_pp']:+.2f}] | "
                     f"{r['p_perm']:.3f} | {r['mean_event_churn']:.3f} |")
    open(os.path.join(RESULTS, "ablations.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    sal = [saliency_vs_true_loss(seed=s) for s in range(3)]
    avg = {c: float(np.mean([x[c] for x in sal])) for c in ("magnitude", "taylor", "fisher", "taylor2")}
    json.dump(dict(per_seed=sal, mean_spearman=avg), open(os.path.join(RESULTS, "saliency_vs_true_loss.json"), "w"), indent=1)
    print("Spearman(score, true loss increase from deleting one weight), mean of 3 seeds:", avg)


if __name__ == "__main__":
    main()
