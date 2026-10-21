"""Part 4: evidence. Sparsity-accuracy sweep, real cost measurement, baseline comparison with
seed-variance analysis, and one falsifiable claim (all numbers computed here, then written to
results/ -- nothing in the README is typed in by hand without a matching file).

    python -m train.run_part4            # full: 7 sparsities x 5 criteria x 10 seeds (~10 min)
    python -m train.run_part4 --quick    # smoke test: 3 seeds, fewer settings

Outputs (results/):
  pareto_raw.csv / pareto_summary.csv / pareto.png      Part 4.1
  cost.csv / cost.json / cost.png                       Part 4.2
  baseline_comparison.csv / baseline_comparison.md      Part 4.3
  claim.md                                              Part 4.4
"""
from . import parallel  # noqa: F401  (must be first: pins BLAS threads before numpy loads)

import argparse
import csv
import json
import os
import time

import numpy as np

from engine import Tensor
from prune import SparseMLP, compact_dense_forward, dense_masked_forward
from .experiment import run_one, HIDDEN
from .parallel import run_many
from .run_part2 import RESULTS
from .stats import paired_stats

METHODS = ["magnitude", "taylor", "fisher", "taylor2", "random"]
PRIMARY = "fisher"
BASELINE = "magnitude"


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ 4.1 sweep
def sweep(sparsities, seeds):
    jobs = [dict(criterion=None, sparsity=0.0, seed=s) for s in seeds]
    jobs += [dict(criterion=m, sparsity=sp, seed=s) for sp in sparsities if sp > 0 for m in METHODS for s in seeds]
    return run_many(jobs, desc="sweep")


def summarize(raw, sparsities, seeds):
    rows = []
    dense = [r["test_acc"] for r in raw if r["criterion"] == "dense"]
    for m in ["dense"] + METHODS:
        for sp in ([0.0] if m == "dense" else [s for s in sparsities if s > 0]):
            rs = [r for r in raw if r["criterion"] == m and r["target_sparsity"] == sp]
            a = np.array([r["test_acc"] for r in rs])
            rows.append(dict(method=m, target_sparsity=sp,
                             achieved_sparsity=float(np.mean([r["achieved_sparsity"] for r in rs])),
                             n_seeds=len(a), test_acc_mean=a.mean(), test_acc_std=a.std(ddof=1),
                             test_acc_min=a.min(), test_acc_max=a.max(),
                             retained_vs_dense=a.mean() / np.mean(dense)))
    return rows


def plot_pareto(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    dense = [r for r in summary if r["method"] == "dense"][0]
    for ax, zoom in zip(axes, (False, True)):
        for i, m in enumerate(METHODS):
            rows = sorted([r for r in summary if r["method"] == m], key=lambda r: r["achieved_sparsity"])
            xs = [0.0] + [r["achieved_sparsity"] for r in rows]
            mu = np.array([dense["test_acc_mean"]] + [r["test_acc_mean"] for r in rows])
            sd = np.array([dense["test_acc_std"]] + [r["test_acc_std"] for r in rows])
            ax.errorbar(xs, mu, yerr=sd, marker="o", ms=4, capsize=2, label=m, color=f"C{i}",
                        lw=2.2 if m == PRIMARY else 1.2)
        ax.axhline(dense["test_acc_mean"], ls="--", color="gray", lw=0.8)
        ax.set(xlabel="achieved sparsity (fraction of weights removed)", ylabel="test accuracy")
        if zoom:
            ax.set(xlim=(0.84, 0.995), ylim=(0.80, 1.0), title="zoom: high sparsity")
        else:
            ax.set(title=f"Sparsity-accuracy Pareto (mean +/- sd over seeds)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# ------------------------------------------------------------------- 4.2 cost
def _time(fn, x, reps):
    fn(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def cost_study(sparsities, seed=0):
    rows = []
    for sp in sparsities:
        res, model, hist, pruner, data = run_one(PRIMARY if sp > 0 else None, sp, seed, return_model=True)
        sparse = SparseMLP(model)
        cfwd, kept, compact_w = compact_dense_forward(model)
        total_w = sum(l.weight.size for l in model.linears())
        for batch in (1, 64, 450):
            x = data.x_test[:batch] if batch <= len(data.x_test) else data.x_test
            ref = model(Tensor(x)).data
            sparse.reset_counters()
            out = sparse.forward(x)
            assert np.allclose(out, ref, atol=1e-9), "sparse forward != dense forward"
            assert np.allclose(cfwd(x), ref, atol=1e-8), "compact forward != dense forward"
            assert np.array_equal(out.argmax(1), ref.argmax(1))
            measured = sparse.mults()
            dense_mults = total_w * len(x)
            reps = 300 if batch < 100 else 60
            t_dense = _time(lambda z: dense_masked_forward(model, z), x, reps)
            t_sparse = _time(sparse.forward, x, reps)
            t_compact = _time(cfwd, x, reps)
            rows.append(dict(
                target_sparsity=sp, batch=len(x), live_weights=sparse.nnz(), total_weights=total_w,
                dense_mults=dense_mults, sparse_mults_measured=measured,
                mult_reduction=dense_mults / max(1, measured), compact_hidden_units=str(kept),
                compact_dense_weights=int(compact_w), us_dense_times_zero=1e6 * t_dense,
                us_sparse_numpy_gather=1e6 * t_sparse, us_compact_dense=1e6 * t_compact,
                sparse_speedup_vs_dense=t_dense / t_sparse, compact_speedup_vs_dense=t_dense / t_compact,
                test_acc=res["test_acc"]))
        print(f"  sparsity {sp:.0%}: {sparse.nnz()}/{total_w} live weights, "
              f"{rows[-1]['mult_reduction']:.1f}x fewer multiplies (measured)", flush=True)
    return rows


def plot_cost(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for b in sorted({r["batch"] for r in rows}):
        rs = [r for r in rows if r["batch"] == b]
        ax[0].plot([r["target_sparsity"] for r in rs], [r["sparse_speedup_vs_dense"] for r in rs], "o-",
                   label=f"sparse NumPy gather, batch {b}")
        ax[1].plot([r["target_sparsity"] for r in rs], [r["compact_speedup_vs_dense"] for r in rs], "o-",
                   label=f"dead-unit compaction, batch {b}")
    ax[0].plot([r["target_sparsity"] for r in rows if r["batch"] == 64],
               [r["mult_reduction"] for r in rows if r["batch"] == 64], "k--", label="multiplies saved (measured)")
    ax[0].set_yscale("log")
    for a, t in zip(ax, ("sparse kernel: FLOPs vs wall-clock", "compaction")):
        a.axhline(1, color="gray", lw=0.8)
        a.set(xlabel="sparsity", ylabel="speed-up vs dense x zeros (x)", title=t)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# ------------------------------------------------------ 4.3 / 4.4 comparisons
def compare(raw, sparsities, seeds):
    rows = []
    for sp in [s for s in sparsities if s > 0]:
        def acc(m):
            d = {r["seed"]: r["test_acc"] for r in raw if r["criterion"] == m and r["target_sparsity"] == sp}
            return np.array([d[s] for s in seeds])

        base = acc(BASELINE)
        for m in METHODS:
            if m == BASELINE:
                continue
            st = paired_stats(acc(m) - base)
            rows.append(dict(sparsity=sp, method=m, baseline=BASELINE, mean_acc_method=acc(m).mean(),
                             mean_acc_baseline=base.mean(), diff_mean=st["mean"], diff_sd=st["sd"],
                             ci95_lo=st["ci_lo"], ci95_hi=st["ci_hi"], p_perm=st["p_perm"], dz=st["dz"],
                             wins=st["wins"], ties=st["ties"], losses=st["losses"], n=st["n"]))
    return rows


def write_markdown(rows, path):
    lines = ["| sparsity | method | acc(method) | acc(magnitude) | diff (pp) | 95% CI (pp) | p (perm.) | W/T/L |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sparsity']:.0%} | {r['method']} | {r['mean_acc_method']:.4f} | "
                     f"{r['mean_acc_baseline']:.4f} | {100 * r['diff_mean']:+.2f} | "
                     f"[{100 * r['ci95_lo']:+.2f}, {100 * r['ci95_hi']:+.2f}] | {r['p_perm']:.4f} | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} |")
    open(path, "w").write("\n".join(lines) + "\n")


def write_claim(raw, summary, comp, seeds, path):
    sp = max(r["sparsity"] for r in comp)
    c = [r for r in comp if r["sparsity"] == sp and r["method"] == PRIMARY][0]
    dense = [r for r in summary if r["method"] == "dense"][0]["test_acc_mean"]
    live = int(round((1 - sp) * 50432))
    low = [r for r in comp if r["sparsity"] <= 0.9 and r["method"] == PRIMARY]
    sig_low = [r for r in low if r["ci95_lo"] > 0 or r["ci95_hi"] < 0]
    low_txt = ("At <= 90% sparsity the paired CI for %s vs magnitude includes 0 at every tested level, i.e. "
               "no detectable difference there." % PRIMARY if not sig_low else
               "At <= 90% sparsity the CI excludes 0 at: " + ", ".join(f"{r['sparsity']:.0%}" for r in sig_low) + ".")
    txt = (f"# Falsifiable claim (auto-generated from results/pareto_raw.csv)\n\n"
           f"**At {sp:.0%} sparsity (~{live} of 50,432 weights alive) on sklearn digits with a "
           f"64-256-128-10 MLP, self-pruning with the `{PRIMARY}` (diagonal empirical Fisher / OBD) "
           f"criterion reaches {100 * c['mean_acc_method']:.2f}% test accuracy versus "
           f"{100 * c['mean_acc_baseline']:.2f}% for magnitude pruning under an identical cubic schedule, "
           f"averaged over {c['n']} seeds (paired difference {100 * c['diff_mean']:+.2f} pp, "
           f"95% bootstrap CI [{100 * c['ci95_lo']:+.2f}, {100 * c['ci95_hi']:+.2f}] pp, "
           f"exact sign-flip permutation p = {c['p_perm']:.4f}, wins/ties/losses = "
           f"{c['wins']}/{c['ties']}/{c['losses']}). "
           f"The dense model scores {100 * dense:.2f}%.**\n\n"
           f"It is falsified if re-running `python -m train.run_part4` yields a paired difference whose "
           f"95% CI includes 0 at this sparsity.\n\n"
           f"Scope / honesty notes:\n"
           f"* The claim is stated at the HIGHEST sparsity in the sweep. That choice is not blind: an "
           f"exploratory 4-seed run during development showed the criteria separating only at very high "
           f"sparsity, so the 98-99% settings are where a difference was expected. The full table over all "
           f"sparsities and all criteria (baseline_comparison.md) is the unfiltered record.\n"
           f"* {low_txt}\n"
           f"* Seeds vary init, mini-batch order AND the train/val/test split, but the dataset is small "
           f"(450 test images: one image = 0.22 pp), and seeds, not test images, are the unit of "
           f"replication.\n")
    open(path, "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    sparsities = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99] if not a.quick else [0.0, 0.9, 0.98]
    seeds = list(range(10)) if not a.quick else [0, 1, 2]
    t0 = time.time()

    print("[4.1] sparsity sweep")
    raw = sweep(sparsities, seeds)
    write_csv(os.path.join(RESULTS, "pareto_raw.csv"), raw)
    summary = summarize(raw, sparsities, seeds)
    write_csv(os.path.join(RESULTS, "pareto_summary.csv"), summary)
    plot_pareto(summary, os.path.join(RESULTS, "pareto.png"))
    for r in summary:
        print(f"  {r['method']:9s} target {r['target_sparsity']:.2f} achieved {r['achieved_sparsity']:.4f} "
              f"acc {r['test_acc_mean']:.4f} +/- {r['test_acc_std']:.4f}")

    print("[4.3] baseline comparison (paired over seeds)")
    comp = compare(raw, sparsities, seeds)
    write_csv(os.path.join(RESULTS, "baseline_comparison.csv"), comp)
    write_markdown(comp, os.path.join(RESULTS, "baseline_comparison.md"))
    print(open(os.path.join(RESULTS, "baseline_comparison.md")).read())

    print("[4.2] cost measurement")
    cost = cost_study([s for s in sparsities], seed=0)
    write_csv(os.path.join(RESULTS, "cost.csv"), cost)
    json.dump(cost, open(os.path.join(RESULTS, "cost.json"), "w"), indent=1)
    plot_cost(cost, os.path.join(RESULTS, "cost.png"))

    print("[4.4] claim")
    write_claim(raw, summary, comp, seeds, os.path.join(RESULTS, "claim.md"))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
