# DESIGN.md

Everything below is backed by code in this repository and numbers in `results/`. Where a result
contradicted what I expected, I say so.

---

## 1. The importance criterion, and why it approximates "loss increase if removed"

**Goal.** For weight `w_i`, importance should be `I_i = L(w with w_i := 0) − L(w)`.

Move along the single coordinate `δ = −w_i` and expand the loss:

```
I_i = −g_i w_i  +  ½ H_ii w_i²  +  O(w_i³),        g = ∂L/∂w,  H = ∂²L/∂w²
```

(The cross terms `H_ij w_i w_j` do not appear because only one weight changes.)
Every criterion in `prune/criteria.py` is a different approximation of this expression.

| criterion | what it keeps of the expansion | assumption it makes |
|---|---|---|
| `magnitude`  `\|w\|` | `w_i²` only | `g_i ≈ 0` **and** `H_ii` equal for all weights |
| `taylor`  `EMA\|w g\|` | first-order term, absolute value | `H_ii w_i²` negligible; sign of a noisy minibatch `g` unreliable → take `\|·\|` and average |
| `fisher`  `½ w² F_ii` | second-order term (Optimal Brain Damage) | `H_ii ≈ F_ii = E_n[g_{n,i}²]` (diagonal empirical Fisher), `g ≈ 0` |
| `taylor2` | both terms | as above |
| `movement` `EMA(−w g)` | Sanh et al. | weights moving away from 0 matter |
| `random` | nothing | sanity floor |

**Why magnitude is the wrong default.** For `z_j = Σ_i x_i W_ij`, `∂²L/∂W_ij² ∝ x_i² · (curvature wrt z_j)`.
The curvature of a weight scales with the **squared activation** feeding it. A large weight on an input
that is almost always ~0 is cheap to delete; a small weight on a high-variance input is not. Magnitude pruning
assumes `H_ii` is constant and cannot see this.

**Why Fisher ≈ Hessian.** For cross-entropy, `H = Gauss-Newton + (residual · ∂²z/∂w²)`. The residual term
vanishes as the model fits, and the Gauss-Newton matrix equals the Fisher `E[∇log p ∇log pᵀ]`. Pruning happens
late in training, when this approximation is good.

**The trap I deliberately avoided.** `F_ii` needs the *per-sample* gradient squared. The square of the
minibatch-mean gradient, `(mean_n g_n)²`, is biased low by `Var_n(g_n)/B` and tends to 0 at convergence — exactly
when you prune. For a Linear layer the per-sample weight gradient is an outer product `x_n δ_nᵀ`, so the exact
diagonal Fisher is one matmul: with a mean loss, `out_grad[n] = δ_n / B`, giving
`F = B · (X∘X)ᵀ (out_grad∘out_grad)` (`prune.criteria.fisher_diag`). `tests/test_pruning.py` checks it against
brute-force per-sample backward passes (equal to 1e-9) and asserts the batch-gradient shortcut underestimates it
by more than 3×.

**Does the derivation hold empirically?** `results/saliency_vs_true_loss.json`: on a trained dense net I
deleted 400 random weights one at a time, measured the true training-loss increase, and rank-correlated it with each
score (mean over 3 seeds):

| magnitude | taylor | fisher | taylor2 |
|---|---|---|---|
| 0.591 | 0.792 | 0.791 | 0.792 |

So gradient information is a real improvement over magnitude (0.59 → 0.79). **But the second-order (Fisher) term adds
nothing measurable over the first-order one**, here or in accuracy (`fisher` vs `taylor`: 95.02 vs 95.33% at 99%,
CIs overlap completely). My prior was that curvature would help; the data say it doesn't at this scale. Since
`taylor` needs no extra matmul (profile below: Fisher bookkeeping is ~13% of a training step), **in production I
would ship `taylor`**. I kept `fisher` as the headline criterion because it was fixed before the experiments, and I report
the others alongside.

**Honest negative result: `movement` collapses** (23% accuracy at 98%, `results/ablations.md`). As implemented
(EMA of `−w·g` over a from-scratch run, with weights re-zeroed on prune) it is a poor ranking signal; the
original method is designed for fine-tuning with a learned score. I report it rather than drop it.

### Schedule

Cubic ramp `s(t) = s_f + (s_i − s_f)(1 − (t−t₀)/(T))³` (Zhu & Gupta). It removes most weights early, while the
net is over-parameterised and removals are cheap, then slows so the importance statistics and the optimizer can
re-route before the budget is hit. Evidence at 98% (10 seeds, paired): cubic 96.36%, linear 96.07% (−0.29 pp,
CI [−0.73, +0.22], not distinguishable), **one-shot 63.2% (−33 pp)**: gradual pruning matters a lot, the cubic-vs-linear choice
does not (at this size).

### Regrowth (bonus) and its stability

Implemented RigL-style (`SelfPruner._update_layer`): per event, keep the top `K−r` live weights by importance,
drop the rest, then revive `r` previously-dead slots with the largest EMA of the **dense** gradient
`|∂L/∂W_eff|`, initialised to exact 0. Stability devices: `r` decays with a cosine to 0 by the end of the
pruning window (the final mask is frozen for fine-tuning); newborn weights are protected from being dropped for
one interval, otherwise a zero-initialised revival has the lowest importance (`w=0`) and is deleted at the next event
(a churn loop); the budget is exact after every event. Measured churn/event (fraction of live weights changed):
0.056 (no regrowth), 0.144 (10%), 0.305 (30%).
**Result: regrowth made no detectable difference to accuracy** (+0.04 and +0.16 pp, CIs include 0). I'm not
claiming a benefit.

---

## 2. What does the engine compute as "the gradient of a masked weight"?

Let `W_eff = M ⊙ W` with a constant 0/1 mask `M`, and `L = f(W_eff)`. There are **two different objects** and
conflating them is the usual bug.

1. **The gradient of the parameter** (what the optimizer sees):

   ```
   ∂L/∂W_ij = M_ij · ∂L/∂(W_eff)_ij        →   exactly 0.0 where M_ij = 0
   ```
   This is the true chain-rule derivative of the function the network computes: the loss does not depend on a masked
   entry, so its derivative is 0 (a finite-difference check in `tests/test_masked.py` agrees). Anything non-zero
   would be the gradient of a *different* function (a straight-through estimator), which makes pruned weights
   drift and pollutes optimizer state.

2. **The dense "what-if" gradient** `G_ij = ∂L/∂(W_eff)_ij`, non-zero at masked entries. It is the gradient the loss
   *would* have if the connection existed: reviving the weight to value `ε` changes the loss by `≈ G_ij ε`. The `masked`
   op does not throw it away (it lives on the `W_eff` node as `.grad`); it is used **only** to rank regrowth
   candidates, never applied as an update. `test_dense_whatif_gradient_is_nonzero_on_masked_and_correct`
   verifies it against finite differences on the effective weight.

**Optimizer state is where implementations go wrong.** Adam with `g = 0` is *not* a no-op: the stored momentum
keeps moving the weight (`m̂/(√v̂+ε) ≠ 0`), and a revived connection inherits stale `m, v` and a global step
counter. Rules in `nn/optim.py`:

* gradient `×M`; moments updated only on live entries and held at **exactly 0** on masked ones; stored weights
  at masked entries are exactly `0.0` after every step (tests assert bit-exact equality);
* Adam uses a **per-element step counter** `t_ij` for bias correction, reset on every prune/revive
  (`on_mask_change`). A revived connection is then a brand-new parameter. Derivation: fresh state, `t=1`,
  `m = (1−β₁)g`, `v = (1−β₂)g²` → `m̂ = g`, `v̂ = g²` → step `= lr·g/(|g|+ε) ≈ lr·sign(g)`. With stale state the step is
  mis-scaled: the textbook implementation measures **0.13×lr** in my test (stale `v`); without any bias correction it
  would be `lr(1−β₁)/√(1−β₂) = 3.16×lr`. `test_revived_connection_first_adam_step_is_exactly_lr` asserts the former
  and `test_naive_adam_shows_both_bugs_that_ours_fixes` demonstrates the latter on a reference naive Adam.

**Honest note on end-to-end impact.** In the ablation "regrowth 30% with NO optimizer-state reset" accuracy is
indistinguishable from the correct version (+0.36 pp, CI [−0.07, +0.76]). Our Adam already zeroes moments of masked
entries every step, so what is left uncorrected is only the step-counter/first-step scale. The correctness is proven
at unit level; I have **not** shown it matters for final accuracy at this scale.

---

## 3. Where does the engine bottleneck, and how would I optimise it?

Measured (single training step, batch 64, 64-256-128-10, float64, multi-threaded BLAS):

| phase | ms | share |
|---|---|---|
| forward (graph build + matmuls) | 1.41 | 19% |
| backward | 2.35 | 32% |
| Adam step | 3.58 | **49%** |
| (+ Fisher bookkeeping when pruning) | +0.94 | +13% |

1. **The optimizer is the biggest single cost**, not the matmuls: ~10 full-size temporaries per parameter per step
   (`np.where`, `**`, divisions). Fix: in-place fused update (`out=` buffers, one pass), float32 state, or a
   compiled kernel (numba/C) doing the whole Adam update + mask in a single loop.
2. **Python-level graph**: one closure and a few array allocations per op, rebuilt every step, plus a
   Python topological sort. Fix: trace once, replay a static tape with preallocated buffers; fuse `matmul+bias(+act)`
   into one op (also removes the `(B,out)` broadcast-add temporaries and halves graph nodes).
3. **Memory**: every node keeps its `.grad` after backward (needed by the pruner, wasteful otherwise). Fix: free
   intermediates after use; keep grads only on tensors the pruner requests.
4. **Masking** re-multiplies `W*M` every forward (50k floats). Fix: cache `W_eff` and refresh only on mask change or
   after the optimizer step; or maintain `W` with exact zeros and skip the multiply in the forward (keeping the multiply in
   the backward for the `M⊙` gradient).
5. **float64 everywhere** (needed for gradcheck, not for training): float32/bfloat16 halves bandwidth.
6. Importance statistics: Fisher costs one extra `(in,B)@(B,out)` matmul per layer. Compute it every k steps, or use
   `taylor`, which costs nothing extra.

---

## 4. Serving a self-pruned model in a multi-tenant inference service at thousands of requests per second

**What I measured about the cost story (`results/cost.csv`).** Multiplies drop exactly `1/(1−s)` (10× at 90%,
~100× at 99%; counted from the kernel, not computed from sparsity). **Wall-clock is a different story**:
my pure-NumPy gather kernel is *slower* than dense BLAS-times-zeros until ~99% sparsity (e.g. 90%, batch 64:
1616 µs vs 179 µs), because BLAS runs near peak on zeros and an unstructured gather is memory-bound.
The wins that did materialise were **dead-unit compaction** (units with no live in/out weight are
removed and replaced by smaller dense matmuls; identical outputs, tested): 1.7× at 90%, 2.8× at 95%, **6.8× at 99%** (batch 64).
The dense-times-zero path is kept only as the comparison; it saves nothing.

Design for production, in order of importance:

1. **Export a frozen inference artifact, not the training object.** No autodiff graph, no masks at runtime:
   compacted dense layers or CSR/blocked-sparse tensors, an integrity hash and the `nnz`/sparsity invariant, version
   tag. Weights are immutable and mmap-shared across workers (copy-on-write free, one copy per host).
2. **Make the sparsity hardware-friendly at train time.** Unstructured sparsity needs a real SpMM kernel
   (C++/SIMD, or GPU block-sparse) to beat dense; I'd constrain the pruner to blocks (e.g. 1×16 / 4×4) or N:M patterns
   so the kernel hits peak, trading a little accuracy (measured with the same Pareto harness) for actual latency.
   Structured/neuron-level pruning is the safest first target: it needs no special kernel.
3. **Dynamic micro-batching per model.** Measured per-sample cost falls from ~12 µs at batch 1 to ~2.4 µs at batch 450;
   at thousands of rps, a 1–5 ms batching window per model buys ~3–5× throughput with a bounded p99.
4. **Multi-tenancy.** A pruned model here is ~0.5–25k live weights (KBs–100 KB): thousands of per-tenant models fit
   in RAM, so use an LRU model registry with lazy load, per-model batching queues, per-tenant concurrency quotas and
   rate limits (noisy-neighbour isolation), and shard by model id (consistent hashing) so hot models stay cache-resident.
5. **Safe rollout.** Shadow/canary the pruned model against the dense one on live traffic, gate on agreement rate and
   latency, keep the dense model as automatic fallback. Monitor logit/entropy drift, NaN/Inf counts, p50/p99 per tenant, and
   re-check the artifact hash on load.
6. **Quantisation composes with pruning** (int8 values + int16 indices): another 2–4× in memory bandwidth, the
   actual bottleneck for sparse inference; validate with the same accuracy harness.

---

## 5. Limitations (what I would not claim)

* Small data (sklearn digits, 450 test images; one image = 0.22 pp). Seeds, not images, are the replication unit,
  and n = 10. I did not run MNIST/CIFAR.
* The headline 99% gap (+10.4 pp for `fisher` vs `magnitude`) is mostly an **allocation artifact** of uniform per-layer
  sparsity that cripples magnitude pruning's 1,280-weight output layer. With global ranking the same comparison is
  95.96% vs 96.62% (+0.67 pp, CI [+0.02, +1.31], p≈0.11). `results/scope_robustness.md`.
* ~24 hypothesis tests in `baseline_comparison.md` are uncorrected; the isolated `p<0.05` entries at 50-75% sparsity
  would not survive a multiple-comparison correction and I do not rely on them.
* The claim sparsity (99%) was selected after an exploratory 4-seed look; stated in `results/claim.md`.
* Fisher ≈ Taylor here; regrowth and the optimizer-state reset show no end-to-end accuracy effect at this scale.
