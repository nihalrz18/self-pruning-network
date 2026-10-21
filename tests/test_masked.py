"""Part 1 correctness requirement: masked weights, their gradients, and optimizer state.

What is asserted (and why it is the right behaviour):
  A. FORWARD   a masked weight contributes an exact 0, whatever value is stored there.
  B. GRADIENT  dL/dW (the parameter) is exactly 0 at masked entries -- verified against
               finite differences, which see a loss that truly does not depend on them.
               The dense "what-if" gradient dL/dW_eff is NON-zero there and equals the
               finite-difference derivative w.r.t. a hypothetically revived weight.
  C. OPTIMIZER pruned entries never move and carry exactly-zero moments (no momentum leak).
  D. REVIVAL   a revived connection starts with pristine state: its first Adam step has
               magnitude lr (bias-corrected, per-element clock). The textbook Adam (global
               step counter, stale moments) does NOT, which we show side by side.
"""
import numpy as np

import engine as E
from engine import Tensor, numerical_grad
from nn import Linear, Adam, SGD, build_mlp

rng = np.random.RandomState(7)


def make_layer(mask_prob=0.5, seed=0):
    r = np.random.RandomState(seed)
    layer = Linear(5, 4, r)
    mask = (r.rand(5, 4) > mask_prob).astype(float)
    layer.set_mask(mask)
    return layer, mask


X = rng.randn(8, 5)
Y = rng.randint(0, 4, size=8)


def loss_of(layer):
    return E.softmax_cross_entropy(layer(Tensor(X)), Y)


# ------------------------------------------------------------------ A. forward
def test_masked_weight_contributes_exactly_zero_to_forward():
    layer, mask = make_layer()
    ref = layer(Tensor(X)).data.copy()
    layer.weight.data[mask == 0] = 1e6  # garbage hidden under the mask
    out = layer(Tensor(X)).data
    np.testing.assert_array_equal(out, ref)  # bit-exact, not "close"
    expected = X @ (layer.weight.data * mask) + layer.bias.data
    np.testing.assert_array_equal(out, expected)


def test_set_mask_makes_stored_weights_true_zeros():
    layer, mask = make_layer()
    assert np.all(layer.weight.data[mask == 0] == 0.0)  # exact 0.0, not "small"


# ----------------------------------------------------------------- B. gradient
def test_parameter_gradient_is_zero_on_masked_and_matches_finite_differences():
    layer, mask = make_layer()
    loss = loss_of(layer)
    loss.backward()
    g = layer.weight.grad
    assert np.all(g[mask == 0] == 0.0)

    def f(w):
        old = layer.weight.data.copy()
        layer.weight.data = w
        val = float(loss_of(layer).data)
        layer.weight.data = old
        return val

    num = numerical_grad(f, layer.weight.data.copy())
    np.testing.assert_allclose(g, num, atol=1e-7, rtol=1e-5)  # also 0 on masked (loss is flat)
    assert np.all(np.abs(num[mask == 0]) < 1e-9)


def test_dense_whatif_gradient_is_nonzero_on_masked_and_correct():
    layer, mask = make_layer()
    loss_of(layer).backward()
    dense = layer.w_eff.grad
    assert np.any(np.abs(dense[mask == 0]) > 1e-6)  # information regrowth can use

    # FD derivative of the loss w.r.t. the effective weight itself (a revived entry would
    # start at 0 and move along exactly this direction)
    w_eff0 = layer.weight.data * mask

    def f(w):
        out = Tensor(X) @ Tensor(w) + layer.bias
        return float(E.softmax_cross_entropy(out, Y).data)

    np.testing.assert_allclose(dense, numerical_grad(f, w_eff0), atol=1e-7, rtol=1e-5)


def test_masked_op_gradcheck():
    m = (rng.rand(3, 4) > 0.5).astype(float)
    E.check_grad(lambda w: E.masked(w, m) * 2.0, rng.randn(3, 4))


# ---------------------------------------------------------------- C. optimizer
def _train_steps(layer, opt, n):
    for _ in range(n):
        opt.zero_grad()
        loss_of(layer).backward()
        opt.step()


def test_adam_keeps_pruned_weights_exactly_zero_with_zero_state():
    layer, mask = make_layer(0.6)
    opt = Adam(layer.parameters(), lr=1e-2)
    _train_steps(layer, opt, 50)
    dead = mask == 0
    w = layer.weight
    assert np.all(w.data[dead] == 0.0)
    assert np.all(opt.m[id(w)][dead] == 0.0) and np.all(opt.v[id(w)][dead] == 0.0)
    assert np.all(opt.t[id(w)][dead] == 0.0)
    assert np.all(opt.t[id(w)][~dead] == 50.0)


def test_sgd_momentum_pruned_entries_have_zero_momentum():
    layer, mask = make_layer(0.6)
    opt = SGD(layer.parameters(), lr=0.05, momentum=0.9)
    _train_steps(layer, opt, 30)
    dead = mask == 0
    assert np.all(layer.weight.data[dead] == 0.0)
    assert np.all(opt.buf[id(layer.weight)][dead] == 0.0)


def test_prune_after_training_wipes_state_and_stops_motion():
    layer, _ = make_layer(0.0)  # fully dense
    opt = Adam(layer.parameters(), lr=1e-2)
    _train_steps(layer, opt, 20)  # build up non-trivial moments
    w = layer.weight
    assert np.abs(opt.m[id(w)]).max() > 0
    new_mask = (rng.rand(5, 4) > 0.5).astype(float)
    pruned = (w.mask > 0) & (new_mask == 0)
    layer.set_mask(new_mask)
    opt.on_mask_change(w, pruned)
    _train_steps(layer, opt, 20)
    assert np.all(w.data[new_mask == 0] == 0.0)
    assert np.all(opt.m[id(w)][new_mask == 0] == 0.0)


def test_optimizer_trajectory_independent_of_garbage_under_mask():
    def run(garbage):
        layer, mask = make_layer(0.5, seed=3)
        layer.weight.data[mask == 0] = garbage  # optimizer must overwrite/ignore it
        opt = Adam(layer.parameters(), lr=1e-2)
        _train_steps(layer, opt, 25)
        return layer.weight.data * mask, layer(Tensor(X)).data

    (w0, o0), (w1, o1) = run(0.0), run(1e3)
    np.testing.assert_array_equal(w0, w1)
    np.testing.assert_array_equal(o0, o1)


# ------------------------------------------------------------------ D. revival
class NaiveAdam:
    """The 'obvious' implementation: global step counter, gradient masked, but moments
    and counter left alone when the mask changes. Kept ONLY to demonstrate the bug."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = {id(p): np.zeros_like(p.data) for p in params}
        self.v = {id(p): np.zeros_like(p.data) for p in params}

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad * (p.mask if p.mask is not None else 1.0)
            m, v = self.m[id(p)], self.v[id(p)]
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            p.data = p.data - self.lr * (m / (1 - self.b1 ** self.t)) / (
                np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


def _prune_then_revive(opt_factory, aware):
    layer, _ = make_layer(0.0, seed=5)
    w = layer.weight
    opt = opt_factory(layer.parameters())
    _train_steps(layer, opt, 40)
    # prune entry (0, 0) for 30 steps
    victim = np.zeros((5, 4), bool)
    victim[0, 0] = True
    m = np.ones((5, 4))
    m[0, 0] = 0
    layer.set_mask(m)
    if aware:
        opt.on_mask_change(w, victim)
    _train_steps(layer, opt, 30)
    stored_while_pruned = w.data[0, 0]
    # revive it with a zero init
    m[0, 0] = 1
    layer.set_mask(m)
    w.data[0, 0] = 0.0
    if aware:
        opt.on_mask_change(w, victim)
    opt.zero_grad()
    loss_of(layer).backward()
    g = w.grad[0, 0]
    before = w.data[0, 0]
    opt.step()
    return stored_while_pruned, w.data[0, 0] - before, g


def test_revived_connection_first_adam_step_is_exactly_lr():
    lr = 1e-2
    stored, delta, g = _prune_then_revive(lambda ps: Adam(ps, lr=lr), aware=True)
    assert stored == 0.0
    assert abs(g) > 1e-8
    # bias-corrected first step of a fresh parameter: m_hat/sqrt(v_hat) = g/|g| = sign(g)
    np.testing.assert_allclose(delta, -lr * np.sign(g), rtol=1e-6)


def test_naive_adam_shows_both_bugs_that_ours_fixes():
    lr = 1e-2
    stored, delta, g = _prune_then_revive(lambda ps: NaiveAdam(ps, lr=lr), aware=False)
    # Bug 1: while "pruned" the stored weight still drifted away from 0 (momentum leak)
    assert stored != 0.0
    # Bug 2: the revived connection's first step is mis-scaled by stale moments / a global
    # clock (here ~8x too small: 1.3e-3 vs lr = 1e-2). Ours is exactly lr (test above).
    assert abs(delta) < 0.5 * lr
