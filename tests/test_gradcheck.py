"""Part 1: finite-difference gradient checks for every differentiable op."""
import numpy as np
import pytest

import engine as E
from engine import Tensor, check_grad

rng = np.random.RandomState(1234)


def randn(*s):
    return rng.randn(*s)


def pos(*s):
    return rng.rand(*s) + 0.5  # bounded away from 0


# ------------------------------------------------------- elementwise binary ops
@pytest.mark.parametrize("op", [E.add, E.sub, E.mul])
@pytest.mark.parametrize("shapes", [
    ((3, 4), (3, 4)),        # same shape
    ((5, 4), (4,)),          # (N, D) + (D,)   <- the classic bias case
    ((5, 4), (5, 1)),        # column broadcast
    ((4,), (5, 4)),          # reflected broadcast
    ((3, 1), (1, 4)),        # both operands stretched (outer-product style)
    ((2, 3, 4), (4,)),       # 3-D with leading broadcast
    ((3, 4), ()),            # scalar operand
])
def test_binary_broadcasting(op, shapes):
    check_grad(op, randn(*shapes[0]), randn(*shapes[1]))


@pytest.mark.parametrize("shapes", [((3, 4), (3, 4)), ((5, 4), (4,)), ((5, 4), (5, 1)), ((3, 1), (1, 4))])
def test_div(shapes):
    check_grad(E.div, randn(*shapes[0]), pos(*shapes[1]))


def test_neg_and_pow():
    check_grad(E.neg, randn(3, 3))
    check_grad(lambda a: E.power(a, 3.0), randn(3, 3))
    check_grad(lambda a: E.power(a, 0.5), pos(3, 3))


def test_operator_sugar():
    check_grad(lambda a, b: (a * b + a / (b * b + 1.0) - 2.0) ** 2, randn(4, 3), randn(3))
    check_grad(lambda a: 3.0 - a * 2.0 + 1.0 / (a * a + 1.0), randn(4))


def test_reused_tensor_accumulates():
    # a appears 3 times: d/da (a*a + a) = 2a + 1
    a = Tensor(np.array([1.5, -2.0]), requires_grad=True)
    (a * a + a).sum().backward()
    np.testing.assert_allclose(a.grad, 2 * a.data + 1)


def test_diamond_graph_topological_order():
    # x -> (u, v) -> w; a wrong topo order would use a stale/partial grad of x's children
    check_grad(lambda x: ((x * 2.0) * (x + 3.0)).tanh() + (x * x).sigmoid(), randn(5))


# ------------------------------------------------------------------- matmul
@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((1, 4), (4, 1)), ((6, 2), (2, 7))])
def test_matmul(shapes):
    check_grad(E.matmul, randn(*shapes[0]), randn(*shapes[1]))


def test_linear_layer_with_bias_broadcast():
    check_grad(lambda x, w, b: (x @ w + b).relu(), randn(6, 4), randn(4, 3), randn(3))


# ---------------------------------------------------------------- reductions
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (0, True),
                                           (1, True), ((0, 1), False), (-1, False)])
def test_sum(axis, keepdims):
    check_grad(lambda a: E.sum_(a, axis, keepdims), randn(3, 4))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), (-1, False)])
def test_mean(axis, keepdims):
    check_grad(lambda a: E.mean(a, axis, keepdims), randn(3, 4))


def test_sum_3d_multi_axis():
    check_grad(lambda a: E.sum_(a, (0, 2)), randn(2, 3, 4))


def test_reshape():
    check_grad(lambda a: a.reshape(4, 3) * 2.0, randn(3, 4))


# ----------------------------------------------------------- non-linearities
def test_relu():
    check_grad(E.relu, randn(4, 5))


def test_tanh():
    check_grad(E.tanh, randn(4, 5))


def test_sigmoid():
    check_grad(E.sigmoid, randn(4, 5))


def test_sigmoid_extreme_inputs_are_stable():
    s = E.sigmoid(Tensor(np.array([-1000.0, 0.0, 1000.0])))
    assert np.all(np.isfinite(s.data))
    np.testing.assert_allclose(s.data, [0.0, 0.5, 1.0])


def test_gelu():
    check_grad(E.gelu, randn(4, 5))


def test_exp_log():
    check_grad(E.exp, randn(3, 3))
    check_grad(E.log, pos(3, 3))


# ------------------------------------------------------ softmax / cross-entropy
def test_softmax():
    check_grad(E.softmax, randn(4, 6))
    s = E.softmax(Tensor(randn(4, 6)))
    np.testing.assert_allclose(s.data.sum(axis=1), 1.0)


def test_log_softmax():
    check_grad(E.log_softmax, randn(4, 6))


def test_softmax_cross_entropy():
    y = rng.randint(0, 6, size=5)
    check_grad(lambda z: E.softmax_cross_entropy(z, y), randn(5, 6))
    check_grad(lambda z: E.softmax_cross_entropy(z, y, reduction="sum"), randn(5, 6))


def test_softmax_cross_entropy_numerically_stable():
    z = Tensor(np.array([[1000.0, -1000.0, 0.0], [-1e4, 1e4, 3.0]]), requires_grad=True)
    loss = E.softmax_cross_entropy(z, np.array([1, 0]))  # confidently wrong -> huge but finite
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(z.grad))
    # d loss / d logits = (softmax - onehot) / N, rows sum to 0
    np.testing.assert_allclose(z.grad.sum(axis=1), 0.0, atol=1e-12)


def test_xent_matches_unfused_composition():
    z = randn(5, 4)
    y = rng.randint(0, 4, size=5)
    fused = E.softmax_cross_entropy(Tensor(z), y).data
    ls = E.log_softmax(Tensor(z)).data
    np.testing.assert_allclose(fused, -ls[np.arange(5), y].mean())


# ---------------------------------------------------------- composite network
def test_two_layer_network_end_to_end():
    y = rng.randint(0, 3, size=7)

    def net(x, w1, b1, w2, b2):
        h = (x @ w1 + b1).tanh()
        return E.softmax_cross_entropy(h @ w2 + b2, y)

    check_grad(net, randn(7, 5), randn(5, 4) * 0.5, randn(4), randn(4, 3) * 0.5, randn(3))


def test_backward_requires_scalar_or_explicit_grad():
    with pytest.raises(ValueError):
        Tensor(np.ones((2, 2)), requires_grad=True).backward()
