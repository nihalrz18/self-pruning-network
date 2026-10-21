"""Sparsity schedules s(t): fraction of weights that must be masked at step t.

cubic (Zhu & Gupta 2017):  s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t_end - t0))^3
  Why cubic: the derivative ds/dt = 3 (s_f - s_i)(1-u)^2 / T is largest at the start and
  decays to 0 at t_end. Early on the net is over-parameterised, so many weights are
  redundant and removing them is cheap; as the net gets sparse each further removal costs
  more loss, so we slow down and give the optimizer time to re-route (and the importance
  statistics time to converge) before the budget is finally hit. Linear removes at a
  constant rate (too fast at the end); one-shot removes everything at once with no chance
  to recover, which is the weakest option (see the ablation in results/ablations.*).
"""
from __future__ import annotations


def cubic(step: int, t0: int, t_end: int, s_final: float, s_init: float = 0.0) -> float:
    if step <= t0:
        return s_init
    if step >= t_end:
        return s_final
    u = (step - t0) / (t_end - t0)
    return s_final + (s_init - s_final) * (1.0 - u) ** 3


def linear(step: int, t0: int, t_end: int, s_final: float, s_init: float = 0.0) -> float:
    if step <= t0:
        return s_init
    if step >= t_end:
        return s_final
    return s_init + (s_final - s_init) * (step - t0) / (t_end - t0)


def oneshot(step: int, t0: int, t_end: int, s_final: float, s_init: float = 0.0) -> float:
    """Everything at once at t_end (no gradual ramp)."""
    return s_final if step >= t_end else s_init


SCHEDULES = {"cubic": cubic, "linear": linear, "oneshot": oneshot}
