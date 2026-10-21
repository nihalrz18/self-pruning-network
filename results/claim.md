# Falsifiable claim (auto-generated from results/pareto_raw.csv)

**At 99% sparsity (~504 of 50,432 weights alive) on sklearn digits with a 64-256-128-10 MLP, self-pruning with the `fisher` (diagonal empirical Fisher / OBD) criterion reaches 95.02% test accuracy versus 84.64% for magnitude pruning under an identical cubic schedule, averaged over 10 seeds (paired difference +10.38 pp, 95% bootstrap CI [+6.00, +16.04] pp, exact sign-flip permutation p = 0.0020, wins/ties/losses = 10/0/0). The dense model scores 97.42%.**

It is falsified if re-running `python -m train.run_part4` yields a paired difference whose 95% CI includes 0 at this sparsity.

Scope / honesty notes:
* The claim is stated at the HIGHEST sparsity in the sweep. That choice is not blind: an exploratory 4-seed run during development showed the criteria separating only at very high sparsity, so the 98-99% settings are where a difference was expected. The full table over all sparsities and all criteria (baseline_comparison.md) is the unfiltered record.
* At <= 90% sparsity the CI excludes 0 at: 75%.
* Seeds vary init, mini-batch order AND the train/val/test split, but the dataset is small (450 test images: one image = 0.22 pp), and seeds, not test images, are the unit of replication.
