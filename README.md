# AQUA — The Self-Pruning Network

A neural-network library written from scratch in **pure Python + NumPy** (reverse-mode autodiff, layers, Adam,
training loop) and a network that **prunes itself while it trains**, under a hard sparsity budget, with the
cost and accuracy evidence to prove it.

> **Restrictions statement.** No PyTorch / TensorFlow / JAX / Keras / HuggingFace / autodiff / pruning /
> NN library is used anywhere. **scikit-learn is used for exactly one thing: `sklearn.datasets.load_digits()` in
> [`train/data.py`](train/data.py) to load the bundled 8×8 digits images.** No sklearn model, metric, optimizer or
> gradient is used. Matplotlib is used for plots. The autodiff engine, backward pass, masking, pruning, optimizers,
> sparse inference path and training loop are my own code.

Read [`DESIGN.md`](DESIGN.md) for the derivations, the masked-gradient reasoning, bottlenecks, serving, and an
explicit list of limitations.

## Install

```bash
# uv (preferred)
uv venv && uv pip install -r requirements.txt
# or plain pip
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # or: pip install -e ".[dev]"
```

Python ≥ 3.10. Developed on 3.14 / NumPy 2.5.

## The five commands

| What | Command | Time |
|---|---|---|
| Gradient-check **and** masked-weight tests (all 82 tests) | `python -m pytest tests -q` | ~5 s |
| Part 2: dense training + Adam + init study | `python -m train.run_part2` | ~20 s |
| Part 3: one self-pruning run (90% sparsity) | `python -m train.run_part3` | ~25 s |
| Part 4: Pareto sweep + cost + baseline comparison + claim | `python -m train.run_part4` | ~14 min (14 cores) |
| Bonus: ablations + saliency-vs-true-loss check | `python -m train.run_ablations` | ~8 min |

Faster variants: `python -m train.run_part4 --quick` (3 seeds, 3 sparsities, ~2 min);
`python -m train.run_part4 --claim-only` recomputes tables and claim from the committed `results/pareto_raw.csv` in 1 s.
Gradient check only: `python -m pytest tests/test_gradcheck.py -q`; masked-weight test only:
`python -m pytest tests/test_masked.py -q`.

Reproducibility: every run is seeded (`seed` fixes init, batch order and the train/val/test split; the pruner RNG is
derived from it). Checked: the ablation table was identical across two complete runs, and the seed-0 models retrained
serially inside the cost study match the parallel sweep's seed-0 accuracies exactly. Experiments run in worker
processes with 1 BLAS thread each; results could differ in the last bits on a different BLAS/CPU, and wall-clock
numbers in `cost.csv` will of course differ per machine.

## Layout

```
engine/   tensor.py (Tensor, all ops, backward)   gradcheck.py (finite differences)
nn/       layers.py (Linear w/ mask, activations, Sequential)   optim.py (SGD-momentum, Adam; mask-aware)   init.py
prune/    criteria.py (importance)   schedule.py   pruner.py (SelfPruner: prune + regrow)   sparse.py (sparse inference)
train/    data.py  trainer.py  experiment.py  run_part2/3/4.py  run_ablations.py  stats.py  parallel.py
tests/    test_gradcheck.py (59)   test_masked.py (11)   test_pruning.py (12)
results/  every plot, CSV and JSON referenced below (committed)
```

## What was built

**Part 1 — autodiff.** `Tensor` with parents + grad; add/sub/mul/div/neg/pow, matmul, sum/mean (any axes,
keepdims), reshape, ReLU, tanh, sigmoid, GELU, exp/log, stable softmax / log-softmax, and a fused
log-sum-exp softmax-cross-entropy; iterative topological `backward()` with gradient accumulation and
`unbroadcast` (verified for `(N,D)+(D,)`, `(N,1)`, `(3,1)+(1,4)`, scalars, 3-D). `tests/test_gradcheck.py` checks every op against
central finite differences using a random output projection (so transposed or mis-broadcast gradients cannot hide).
**Masked weights:** a dedicated `masked(W, M)` op; the parameter gradient is `M ⊙ ∂L/∂W_eff` (exactly 0 on pruned slots)
and the dense what-if gradient is preserved separately for regrowth. The optimizers keep masked state at exactly 0 and
use a per-element Adam clock reset on prune/revive. `tests/test_masked.py` proves: bit-exact zero contribution to
the forward; zero parameter gradient matching finite differences; correct non-zero dense gradient; no momentum leak; a revived weight's
first Adam step is exactly `lr`; the textbook implementation fails this. See DESIGN.md §2.

**Part 2 — training.** He-initialised MLP (`64-256-128-10`), Adam from scratch (SGD-momentum also provided), own
mini-batch loop. `python -m train.run_part2` → `results/part2_learning_curve.png`, `part2_curve.json`:
dense reaches 100% train / 97.8% test with no NaNs (max grad-norm 2.4). Init justification
(`results/part2_init_study.json`): with `Var(W)=2/fan_in` pre-activation std stays ≈1.4–1.6 across layers; 0.1×He
collapses the signal to 0.002 by layer 3, 10×He explodes it to 1,600 (and trains to only 89%), zeros never learn (8%).

**Part 3 — self-pruning.** `prune/pruner.py`. Importance criteria (magnitude, first-order Taylor, diagonal-Fisher
OBD, both terms, movement, random); cubic sparsity ramp from epoch 10 to 50 (pruning event every 10 steps),
then fine-tuning at a frozen mask; pruned weights are stored as exact `0.0`, masks apply inside the forward; optional
RigL-style regrowth. `python -m train.run_part3` hits **exactly 90.00%** sparsity (per-layer 1638/16384, 3277/32768,
128/1280 live) at **98.9%** test accuracy (seed 0; dense 98.0% on the same split) → `results/part3_run.png`.

**Part 4 — evidence** (10 seeds each; mean test accuracy % ± sd; seeds vary init, batch order and data split):

| target sparsity | live weights | magnitude | taylor | fisher | taylor2 | random |
|---|---|---|---|---|---|---|
| 0% (dense) | 50,432 | 97.42 ± 0.56 | | | | |
| 50% | 25,216 | 97.27 ± 0.65 | 97.60 ± 0.67 | 97.38 ± 0.74 | 97.62 ± 0.76 | 97.20 ± 0.69 |
| 75% | 12,608 | 97.24 ± 0.52 | 97.71 ± 0.57 | 97.62 ± 0.69 | 97.53 ± 0.57 | 96.91 ± 0.61 |
| 90% | 5,043 | 97.33 ± 0.58 | 97.42 ± 0.66 | 97.44 ± 0.79 | 97.53 ± 0.61 | 96.36 ± 0.90 |
| 95% | 2,522 | 97.24 ± 0.65 | 96.96 ± 0.79 | 97.29 ± 0.34 | 97.16 ± 0.92 | 95.60 ± 0.98 |
| 98% | 1,009 | 96.04 ± 1.34 | 96.87 ± 0.67 | 96.36 ± 1.05 | 96.51 ± 1.05 | 69.67 ± 5.95 |
| 99% | 504 | 84.64 ± 8.37 | 95.33 ± 1.06 | 95.02 ± 1.54 | 94.13 ± 2.95 | 27.16 ± 3.43 |

`results/pareto.png`, `pareto_summary.csv`, **raw per-run numbers in `pareto_raw.csv`**.
Achieved sparsity equals target exactly in every run.

1. **Pareto curve:** above. 90% of weights can be removed at no measurable accuracy cost (97.4% vs 97.4% dense); 95% costs <0.2 pp for magnitude/fisher (taylor: −0.5 pp); none of these differences is distinguishable from seed noise.
2. **Real cost** (`results/cost.csv`, `cost.png`; the sparse path is my own CSC gather kernel with a counter on the multiplies actually executed, outputs verified equal to the dense forward):
   multiplies fall exactly `1/(1−s)` — 10.0× at 90%, 99.9× at 99% (measured, not computed).
   **Wall-clock is more nuanced and I report it as measured:** the pure-NumPy unstructured kernel is *slower* than dense BLAS on zeros until ~99% (batch 64: 1616 µs vs 179 µs at 90%; 163 µs vs 183 µs at 99%), because BLAS runs near peak and gather is memory-bound.
   Dead-unit compaction (structured by-product of unstructured pruning, dense matmuls on a smaller net, identical outputs) gives real speed-ups: 1.7× at 90%, 2.8× at 95%, 6.8× at 99% (batch 64). The `dense × zero` shortcut is included only as the baseline: it performs every multiply and is **not** a speed-up.
3. **Baseline comparison** (`results/baseline_comparison.md`, paired by seed, bootstrap CI + exact sign-flip permutation test): at ≤ 95% sparsity the gradient-based criteria are **not reliably better than magnitude** — differences of ±0.1–0.5 pp are within seed noise (sd ≈ 0.6 pp). Random pruning is clearly worse from 90% (−1.0 pp, p = 0.002). The advantage appears when the budget is tight.
4. **One falsifiable claim** (`results/claim.md`, auto-generated from the committed raw numbers):
   *At 99% sparsity (~504 of 50,432 weights) with uniform per-layer sparsity and the same cubic schedule, `fisher` self-pruning reaches **95.02%** test accuracy vs **84.64%** for magnitude pruning (10 seeds; paired difference +10.38 pp, 95% bootstrap CI [+6.00, +16.04], exact permutation p = 0.002, 10/0/0 wins; dense 97.42%).*
   **Read the caveat honestly:** most of that gap is due to uniform per-layer allocation starving the small output layer of magnitude pruning. With global ranking (`results/scope_robustness.md`) the comparison is 95.96% (magnitude) vs 96.62% (fisher), +0.67 pp, CI [+0.02, +1.31], p = 0.11: **not** a reliable win. What the evidence does support robustly: *gradient-aware importance is far more robust than magnitude to a naive/uniform allocation at extreme sparsity, and every informed criterion crushes random.*

**Bonus (`results/ablations.md`, 98% sparsity, 10 seeds, paired):** one-shot pruning collapses (63.2%, −33 pp) while cubic vs linear ramps are indistinguishable; regrowth (10%, 30%) neither helps nor hurts (churn 0.14 / 0.31 per event); skipping the optimizer-state reset on revival is indistinguishable end-to-end at this scale (but fails the unit test); global ranking helps magnitude pruning (+0.7 pp); the `movement` criterion fails (23%) as I implemented it; and a direct test of the Taylor derivation: Spearman correlation with the *true* loss increase from deleting one weight is 0.59 (magnitude) vs 0.79 (taylor / fisher / taylor2).

## Things I want reviewers to know up front

* Fisher ≈ first-order Taylor in every measurement; I would ship Taylor (cheaper). Curvature did not help here.
* The test split is small (450 images); effects below ~0.5 pp are noise. About two dozen uncorrected comparisons are in `baseline_comparison.md`; the lone `p<0.05` entries at 50–75% sparsity are not trusted.
* The claim's sparsity level was chosen after an exploratory look (stated in `claim.md`).
* No MNIST/CIFAR (no network access assumed); digits is small but fully offline and reproducible.
