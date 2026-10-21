"""SelfPruner: prunes (and optionally regrows) connections *during* training.

Lifecycle per training step (driven by train.trainer.fit):
    loss.backward()
    pruner.observe(step)    # update importance statistics from this step's gradients
    optimizer.step()
    pruner.update(step)     # every `every` steps in [start, end]: change the masks

A mask update at step t for layer l:
  1. target number of live weights  K_l = round((1 - s(t)) * N_l)      (s = schedule)
  2. grow budget  r = round(regrow_frac * K_l * cosine_decay(t))       (0 after `end`)
  3. KEEP the top (K_l - r) currently-live weights by importance (recently revived ones are
     protected for `protect` steps so regrowth is not instantly undone => no churn loop)
  4. DROP every other live weight -> mask 0, stored value exactly 0.0
  5. GROW the r not-currently-live slots with the largest EMA |dense gradient| (the
     what-if gradient dL/dW_eff that the masked op still exposes) -> mask 1, value 0.0
  6. wipe optimizer state at every changed slot (`Optimizer.on_mask_change`) and the
     importance statistics there, so a revived weight is a genuinely new parameter.
Scope 'global' lets layers have different sparsities: K_l comes from one global threshold
over all live weights' scores (with a per-layer floor); regrowth then proceeds per layer.
"""
from __future__ import annotations

import math

import numpy as np

from .criteria import CRITERIA, LayerStats, importance, tracked_stats
from .schedule import SCHEDULES


def _topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask with exactly k True entries at the largest scores (ties by index)."""
    flat = score.ravel()
    out = np.zeros(flat.size, bool)
    if k > 0:
        out[np.argsort(-flat, kind="stable")[:k]] = True
    return out.reshape(score.shape)


class SelfPruner:
    def __init__(self, criterion: str = "fisher", final_sparsity: float = 0.9,
                 schedule: str = "cubic", start_step: int = 200, end_step: int = 1200,
                 every: int = 25, regrow_frac: float = 0.0, scope: str = "layer",
                 ema: float = 0.9, protect: int | None = None, min_layer_frac: float = 0.02,
                 reset_state: bool = True, seed: int = 0):
        assert criterion in CRITERIA, criterion
        assert scope in ("layer", "global")
        self.criterion, self.s_final = criterion, final_sparsity
        self.sched = SCHEDULES[schedule]
        self.t0, self.t_end, self.every = start_step, end_step, every
        self.regrow_frac, self.scope, self.ema = regrow_frac, scope, ema
        self.protect = every if protect is None else protect
        self.min_layer_frac = min_layer_frac
        self.reset_state = reset_state  # False only for the ablation that reproduces the bug
        self.rng = np.random.RandomState(seed + 777)
        self.log: list[dict] = []
        self.model = None

    # -------------------------------------------------------------- wiring
    def attach(self, model, optimizer):
        self.model, self.opt = model, optimizer
        self.layers = model.linears()
        track = tracked_stats(self.criterion, self.regrow_frac > 0)
        self.stats = [LayerStats(l.weight.shape, track, self.rng) for l in self.layers]
        self.age = [np.full(l.weight.shape, 10 ** 9) for l in self.layers]  # steps since (re)born

    def observe(self, step: int):
        for l, st in zip(self.layers, self.stats):
            st.observe(l, self.ema)
        for a in self.age:
            a += 1

    def target_sparsity(self, step: int) -> float:
        return self.sched(step, self.t0, self.t_end, self.s_final)

    # -------------------------------------------------------------- updates
    def update(self, step: int):
        if step < self.t0 or step > self.t_end:
            return
        if (step - self.t0) % self.every != 0 and step != self.t_end:
            return
        s = self.target_sparsity(step)
        cur = [float(l.mask.mean()) for l in self.layers]  # live fraction
        if s <= 0.0 or all(c <= (1 - s) + 1e-12 for c in cur):
            return
        progress = min(1.0, max(0.0, (step - self.t0) / max(1, self.t_end - self.t0)))
        decay = 0.5 * (1 + math.cos(math.pi * progress)) if step < self.t_end else 0.0
        scores = [importance(self.criterion, l, st) for l, st in zip(self.layers, self.stats)]
        targets = self._layer_targets(s, scores)
        info = dict(step=step, target_sparsity=s, layers=[])
        for i, (l, st) in enumerate(zip(self.layers, self.stats)):
            info["layers"].append(self._update_layer(i, l, st, scores[i], targets[i], decay, step))
        self.log.append(info)

    def _layer_targets(self, s: float, scores: list[np.ndarray]) -> list[int]:
        sizes = [l.mask.size for l in self.layers]
        if self.scope == "layer":
            return [max(1, int(round((1 - s) * n))) for n in sizes]
        total_live = int(round((1 - s) * sum(sizes)))
        pooled = np.concatenate([sc[l.mask > 0].ravel() for sc, l in zip(scores, self.layers)])
        thr = np.sort(pooled)[::-1][min(total_live, len(pooled)) - 1]
        out = []
        for sc, l, n in zip(scores, self.layers, sizes):
            k = int(((sc >= thr) & (l.mask > 0)).sum())
            out.append(max(k, int(self.min_layer_frac * n), 1))
        return out

    def _update_layer(self, i, layer, st, score, k_target, decay, step) -> dict:
        old = layer.mask > 0
        n_live = int(old.sum())
        k_target = min(k_target, n_live)
        r = int(round(self.regrow_frac * k_target * decay))
        r = min(r, k_target // 2, int((~old).sum()))
        k_keep = k_target - r

        s_keep = np.where(old, score, -np.inf)
        s_keep = np.where(old & (self.age[i] < self.protect), np.inf, s_keep)  # protect newborns
        keep = _topk_mask(s_keep, k_keep)

        grow = np.zeros_like(old)
        if r > 0:
            cand = np.where(~old, st.dense_grad, -np.inf)  # only slots dead BEFORE this event
            grow = _topk_mask(cand, r)

        new = keep | grow
        dropped = old & ~new
        grown = grow & ~old
        layer.weight.data[dropped] = 0.0
        layer.set_mask(new.astype(np.float64))
        layer.weight.data[grown] = 0.0  # revived weights start from exact 0 (RigL convention)
        changed = dropped | grown
        if self.reset_state:
            self.opt.on_mask_change(layer.weight, changed)
        st.reset(changed)
        self.age[i][grown] = 0
        return dict(layer=i, live=int(new.sum()), size=int(new.size), dropped=int(dropped.sum()),
                    grown=int(grown.sum()), churn=float(changed.sum() / max(1, old.sum())))
