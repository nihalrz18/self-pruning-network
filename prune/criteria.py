"""Importance criteria: how a connection earns the right to survive.

Goal. Importance of weight w_i should be the loss increase from removing it:
        I_i = L(w with w_i := 0) - L(w).
Expand L around the current weights along the single coordinate  delta = -w_i :
        I_i = -g_i w_i + 1/2 H_ii w_i^2 + O(w_i^3),     g = dL/dw, H = d2L/dw2.

magnitude   I = |w|                     Assumes g_i w_i ~ 0 and H_ii is the SAME for every
                                        weight. H_ii = E[(dz/dw_i)^2 ...] scales with the
                                        squared input activation x_j^2 feeding the weight,
                                        so a weight on a dead / tiny input is cheap to
                                        remove regardless of |w|. Magnitude ignores this.
taylor      I = EMA |w * g|             First-order term. |.| because a single mini-batch
                                        g is noisy (sign unreliable) and at a minimum g->0
                                        so the signed term vanishes; the EMA over steps is
                                        the usual Molchanov et al. estimator.
fisher      I = 1/2 w^2 F_ii            Second-order (Optimal Brain Damage) term with the
            F_ii = EMA E_n[g_{n,i}^2]   diagonal empirical Fisher in place of the Hessian:
                                        for cross-entropy with a well-fit model the Fisher
                                        equals the Gauss-Newton matrix, which is the Hessian
                                        minus a term that vanishes as residuals -> 0.
                                        CRUCIAL: F needs PER-SAMPLE gradients g_n. The square
                                        of the mini-batch mean gradient, (mean_n g_n)^2,
                                        is NOT it -- it is biased low by Var_n(g_n)/B and
                                        goes to 0 at convergence, exactly when pruning
                                        happens. For a linear layer the per-sample gradient
                                        is an outer product x_n delta_n^T, so
                                        F_ij = (1/B) sum_n x_{n,i}^2 delta_{n,j}^2 is one
                                        matmul  (see `fisher_diag`).
taylor2     I = EMA|w g| + 1/2 w^2 F    Both terms of the second-order expansion.
movement    I = EMA of -w * g           Sanh et al.: weights moving away from zero matter.
random      I ~ U(0,1)                  Sanity floor.
"""
from __future__ import annotations

import numpy as np

CRITERIA = ("magnitude", "taylor", "fisher", "taylor2", "movement", "random")


def fisher_diag(x: np.ndarray, out_grad: np.ndarray) -> np.ndarray:
    """Exact per-sample diagonal empirical Fisher of a Linear layer's weight, (in, out).

    The loss is a MEAN over B samples, so out_grad[n] = delta_n / B where delta_n is the
    gradient of sample n's own loss wrt its pre-activation. Per-sample weight gradient is
    g_n = x_n (outer) delta_n, hence
        F = (1/B) sum_n g_n^2 = (1/B) sum_n (x_n^2)(outer)(delta_n^2)
          = B * (X^2)^T (out_grad^2).
    """
    b = x.shape[0]
    return b * (x * x).T @ (out_grad * out_grad)


class LayerStats:
    """Running statistics for one Linear layer (all arrays shaped like its weight)."""

    def __init__(self, shape, track: set[str], rng: np.random.RandomState):
        self.track = track
        self.n_obs = 0
        self.taylor = np.zeros(shape)
        self.fisher = np.zeros(shape)
        self.movement = np.zeros(shape)
        self.dense_grad = np.zeros(shape)  # EMA |dL/dW_eff|: regrowth signal at pruned slots
        self.rand = rng.rand(*shape)

    def observe(self, layer, ema: float):
        w, g = layer.weight.data, layer.weight.grad  # g is exactly 0 where masked
        first = self.n_obs == 0
        upd = (lambda old, new: new) if first else (lambda old, new: ema * old + (1 - ema) * new)
        if "taylor" in self.track:
            self.taylor = upd(self.taylor, np.abs(w * g))
        if "fisher" in self.track:
            self.fisher = upd(self.fisher, fisher_diag(layer.last_input, layer.out.grad))
        if "movement" in self.track:
            self.movement = upd(self.movement, -w * g)
        if "dense_grad" in self.track:
            self.dense_grad = upd(self.dense_grad, np.abs(layer.w_eff.grad))
        self.n_obs += 1

    def reset(self, where: np.ndarray):
        """Forget history at pruned / revived slots: a revived connection is a new one."""
        for name in ("taylor", "movement", "fisher", "dense_grad"):
            getattr(self, name)[where] = 0.0


def tracked_stats(criterion: str, regrow: bool) -> set[str]:
    t = {"taylor": {"taylor"}, "fisher": {"fisher"}, "taylor2": {"taylor", "fisher"},
         "movement": {"movement"}}.get(criterion, set())
    return t | ({"dense_grad"} if regrow else set())


def importance(criterion: str, layer, st: LayerStats) -> np.ndarray:
    """Score for every slot of `layer.weight` (higher = more important)."""
    w = layer.weight.data
    if criterion == "magnitude":
        return np.abs(w)
    if criterion == "taylor":
        return st.taylor
    if criterion == "fisher":
        return 0.5 * w * w * st.fisher
    if criterion == "taylor2":
        return st.taylor + 0.5 * w * w * st.fisher
    if criterion == "movement":
        return st.movement
    if criterion == "random":
        return st.rand
    raise ValueError(criterion)
