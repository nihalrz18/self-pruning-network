# Falsifiable claim (auto-generated from results/pareto_raw.csv)

**At 98% sparsity (~1009 of 50,432 weights alive) on sklearn digits with a 64-256-128-10 MLP, self-pruning with the `fisher` (diagonal empirical Fisher / OBD) criterion reaches 96.96% test accuracy versus 97.26% for magnitude pruning under an identical cubic schedule, averaged over 3 seeds (paired difference -0.30 pp, 95% bootstrap CI [-1.56, +0.89] pp, exact sign-flip permutation p = 0.7500, wins/ties/losses = 1/0/2). The dense model scores 97.48%.**

It is falsified if re-running `python -m train.run_part4` yields a paired difference whose 95% CI includes 0 at this sparsity.

Scope / honesty notes: seeds vary init, mini-batch order AND the train/val/test split, but the dataset is small (450 test images, so one image = 0.22 pp). At <= 90% sparsity the methods are statistically indistinguishable (see baseline_comparison.md): the benefit of sensitivity-aware importance appears only when the budget becomes tight.
