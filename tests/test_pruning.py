"""Part 3 / 4 mechanics: Fisher estimator, schedule, exact masking, regrowth, sparse path."""
import numpy as np
import pytest

import engine as E
from engine import Tensor
from nn import build_mlp, Adam
from prune import SelfPruner, fisher_diag, cubic, linear, oneshot, SparseMLP, compact_dense_forward, \
    dense_masked_forward
from train.data import make_spirals
from train.trainer import fit, global_sparsity


# ------------------------------------------------------------------ Fisher
def test_fisher_diag_matches_bruteforce_per_sample_gradients():
    rng = np.random.RandomState(0)
    model = build_mlp([6, 5, 3], seed=1)
    x, y = rng.randn(9, 6), rng.randint(0, 3, size=9)
    lin = model.linears()[0]
    # one batched backward: out_grad rows are delta_n / B
    model.zero_grad()
    E.softmax_cross_entropy(model(Tensor(x)), y).backward()
    fast = fisher_diag(lin.last_input, lin.out.grad)
    # brute force: per-sample gradient with B=1, then mean of squares
    acc = np.zeros_like(lin.weight.data)
    for n in range(len(x)):
        model.zero_grad()
        E.softmax_cross_entropy(model(Tensor(x[n:n + 1])), y[n:n + 1]).backward()
        acc += lin.weight.grad ** 2
    np.testing.assert_allclose(fast, acc / len(x), rtol=1e-9, atol=1e-14)


def test_minibatch_grad_squared_is_not_the_fisher():
    rng = np.random.RandomState(0)
    model = build_mlp([6, 5, 3], seed=1)
    x, y = rng.randn(32, 6), rng.randint(0, 3, size=32)
    lin = model.linears()[0]
    model.zero_grad()
    E.softmax_cross_entropy(model(Tensor(x)), y).backward()
    fisher = fisher_diag(lin.last_input, lin.out.grad)
    naive = lin.weight.grad ** 2
    assert fisher.sum() > 3 * naive.sum()  # the shortcut underestimates curvature badly


# ---------------------------------------------------------------- schedules
def test_schedules():
    assert cubic(0, 100, 1100, 0.9) == 0.0 and cubic(1100, 100, 1100, 0.9) == 0.9
    assert cubic(50, 100, 1100, 0.9) == 0.0
    vals = [cubic(t, 100, 1100, 0.9) for t in range(0, 1300, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # cubic is concave: front-loaded; halfway through it is past 87% of the final sparsity
    assert cubic(600, 100, 1100, 0.9) == pytest.approx(0.9 * (1 - 0.5 ** 3))
    assert linear(600, 100, 1100, 0.9) == pytest.approx(0.45)
    assert oneshot(1099, 100, 1100, 0.9) == 0.0 and oneshot(1100, 100, 1100, 0.9) == 0.9


# ------------------------------------------------------- pruning end-to-end
def _run(criterion, sparsity, regrow=0.0, epochs=30, seed=0):
    data = make_spirals(200, 3)
    model = build_mlp([2, 64, 64, 3], seed=seed)
    spe = int(np.ceil(len(data.x_train) / 32))
    pr = SelfPruner(criterion, sparsity, start_step=spe * 3, end_step=spe * 20, every=spe // 2 or 1,
                    regrow_frac=regrow, seed=seed)
    hist = fit(model, data, epochs=epochs, batch_size=32, lr=5e-3, seed=seed, pruner=pr)
    return model, hist, pr


@pytest.mark.parametrize("criterion", ["magnitude", "taylor", "fisher", "taylor2", "movement"])
def test_hits_exact_target_sparsity_with_true_zeros(criterion):
    model, hist, pr = _run(criterion, 0.8)
    for l in model.linears():
        live = int(l.mask.sum())
        assert live == max(1, int(round(0.2 * l.mask.size)))
        assert np.all(l.weight.data[l.mask == 0] == 0.0)  # genuine zeros
        assert np.all(np.isfinite(l.weight.data))
    assert global_sparsity(model) == pytest.approx(0.8, abs=0.01)
    assert hist["test_acc"][-1] > 0.6  # still a working classifier


def test_regrowth_keeps_budget_exact_and_resets_state():
    model, hist, pr = _run("fisher", 0.8, regrow=0.3)
    assert any(layer["grown"] > 0 for ev in pr.log for layer in ev["layers"]), "no regrowth happened"
    for l in model.linears():
        assert int(l.mask.sum()) == max(1, int(round(0.2 * l.mask.size)))
        assert np.all(l.weight.data[l.mask == 0] == 0.0)
    assert np.isfinite(hist["train_loss"]).all()


def test_revived_slot_has_zero_weight_and_pristine_adam_state():
    data = make_spirals(100, 3)
    model = build_mlp([2, 16, 3], seed=0)
    pr = SelfPruner("fisher", 0.6, start_step=5, end_step=40, every=5, regrow_frac=0.5)
    opt = Adam(model.parameters(), lr=1e-2)
    pr.attach(model, opt)
    rng = np.random.RandomState(0)
    for step in range(60):
        b = rng.choice(len(data.x_train), 32)
        opt.zero_grad()
        E.softmax_cross_entropy(model(Tensor(data.x_train[b])), data.y_train[b]).backward()
        pr.observe(step)
        opt.step()
        before = [l.mask.copy() for l in model.linears()]
        pr.update(step)
        for l, old in zip(model.linears(), before):
            revived = (l.mask > 0) & (old == 0)
            dropped = (l.mask == 0) & (old > 0)
            w = l.weight
            for arr in (opt.m[id(w)], opt.v[id(w)], opt.t[id(w)]):
                assert np.all(arr[revived] == 0) and np.all(arr[dropped] == 0)
            assert np.all(w.data[revived] == 0) and np.all(w.data[dropped] == 0)


def test_cubic_schedule_is_followed_during_training():
    model, hist, pr = _run("magnitude", 0.8)
    assert hist["sparsity"][0] < 0.2 and hist["sparsity"][-1] == pytest.approx(0.8, abs=0.01)
    assert all(b >= a - 1e-9 for a, b in zip(hist["sparsity"], hist["sparsity"][1:]))


# --------------------------------------------------------- sparse inference
def test_sparse_forward_equals_dense_masked_forward_and_counts_real_multiplies():
    model, _, _ = _run("fisher", 0.9, epochs=25)
    x = np.random.RandomState(3).randn(17, 2)
    ref = model(Tensor(x)).data
    sp = SparseMLP(model)
    np.testing.assert_allclose(sp.forward(x), ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense_masked_forward(model, x), ref, rtol=1e-10, atol=1e-12)
    dense_mults = sum(l.weight.size for l in model.linears()) * 17
    assert sp.mults() == 17 * sp.nnz()            # measured == batch * nnz
    assert sp.mults() < 0.15 * dense_mults        # ~90% fewer multiplies actually executed
    fwd, kept, _ = compact_dense_forward(model)
    np.testing.assert_allclose(fwd(x), ref, rtol=1e-9, atol=1e-10)
