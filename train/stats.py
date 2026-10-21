"""Small paired-comparison statistics (NumPy only)."""
from __future__ import annotations

import itertools

import numpy as np


def paired_stats(diffs, n_boot: int = 10000, seed: int = 0) -> dict:
    """Stats for per-seed paired differences d_s = acc_A(s) - acc_B(s).

    Pairing is legitimate: both methods share the seed, hence the data split, the
    initial weights and the mini-batch order. Seeds are the unit of replication (n =
    len(diffs)); test examples within a seed are NOT independent replicates.

    * 95% CI: percentile bootstrap over seeds.
    * p: exact two-sided sign-flip permutation test of H0 "A and B are exchangeable within
      each seed" (2^n flips for n <= 16, Monte-Carlo otherwise).
    * dz: Cohen's d for paired samples = mean / sd.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    rng = np.random.RandomState(seed)
    boots = d[rng.randint(0, n, size=(n_boot, n))].mean(1)
    if n <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    else:
        signs = rng.choice([-1.0, 1.0], size=(20000, n))
    null = (signs * d).mean(1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())
    sd = d.std(ddof=1) if n > 1 else float("nan")
    return dict(n=n, mean=float(d.mean()), sd=float(sd), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), p_perm=p,
                dz=float(d.mean() / sd) if sd and sd > 0 else float("nan"),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))
