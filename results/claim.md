# Falsifiable claim (auto-generated from results/pareto_raw.csv)

**At 99% sparsity (~504 of 50,432 weights alive) on sklearn digits with a 64-256-128-10 MLP, self-pruning with the `fisher` (diagonal empirical Fisher / OBD) criterion reaches 95.02% test accuracy versus 84.64% for magnitude pruning under an identical cubic schedule and uniform per-layer sparsity, averaged over 10 seeds (paired difference +10.38 pp, 95% bootstrap CI [+6.00, +16.04] pp, exact sign-flip permutation p = 0.0020, wins/ties/losses = 10/0/0). The dense model scores 97.42%.**

It is falsified if re-running `python -m train.run_part4` yields a paired difference whose 95% CI includes 0 at this sparsity.

Scope / honesty notes:
* The claim is stated at the HIGHEST sparsity in the sweep. That choice is not blind: an exploratory 4-seed run during development showed the criteria separating only at very high sparsity, so the 98-99% settings are where a difference was expected. The full table over all sparsities and all criteria (baseline_comparison.md) is the unfiltered record.
* At <= 90% sparsity the CI excludes 0 at: 75%.
* IMPORTANT CAVEAT (from `python -m train.run_ablations`): the 99% gap is largely an allocation effect. With uniform per-layer sparsity the 1,280-weight output layer is cut to ~13 weights, which hurts magnitude pruning most. If layers are instead ranked globally the gap shrinks to the table below.
* Seeds vary init, mini-batch order AND the train/val/test split, but the dataset is small (450 test images: one image = 0.22 pp), and seeds, not test images, are the unit of replication.

Scope robustness at 99% sparsity, 10 seeds. Paired difference = criterion minus magnitude, same scope.

| scope | criterion | test acc (mean +/- sd) | diff vs magnitude (pp) | 95% CI (pp) | p | W/T/L |
|---|---|---|---|---|---|---|
| layer | magnitude | 0.8464 +/- 0.0837 | - | - | - | - |
| layer | taylor | 0.9533 +/- 0.0106 | +10.69 | [+6.33, +16.40] | 0.0020 | 10/0/0 |
| layer | fisher | 0.9502 +/- 0.0154 | +10.38 | [+6.00, +16.04] | 0.0020 | 10/0/0 |
| global | magnitude | 0.9596 +/- 0.0105 | - | - | - | - |
| global | taylor | 0.9653 +/- 0.0073 | +0.58 | [-0.11, +1.36] | 0.2070 | 5/1/4 |
| global | fisher | 0.9662 +/- 0.0110 | +0.67 | [+0.02, +1.31] | 0.1055 | 7/1/2 |
