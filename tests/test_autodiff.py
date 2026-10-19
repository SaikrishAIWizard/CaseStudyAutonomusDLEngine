import numpy as np
import pytest

from autodiff import (
    Tensor,
    apply_mask,
    check_gradient,
    gelu,
    mean,
    relu,
    sigmoid,
    softmax,
    softmax_cross_entropy,
    sum,
    tanh,
)


def test_backward_accumulates_shared_paths_and_repeated_calls() -> None:
    x = Tensor(3.0, requires_grad=True)
    loss = x * x + x

    loss.backward()
    np.testing.assert_allclose(x.grad, 7.0)

    loss.backward()
    np.testing.assert_allclose(x.grad, 14.0)

    loss.zero_grad()
    assert x.grad is None


@pytest.mark.parametrize(
    "fn, values",
    [
        (lambda x: sum(x + np.array([1.0, 2.0, 3.0])), np.arange(6.0).reshape(2, 3)),
        (lambda x: mean(x * np.array([[1.0], [2.0]])), np.arange(6.0).reshape(2, 3)),
        (lambda x: sum(-x), np.array([-0.8, 0.4, 1.2])),
        (lambda x: sum(x**3), np.array([-0.8, 0.4, 1.2])),
        (
            lambda x: sum((x - np.array([1.0, 2.0])) / np.array([[2.0], [3.0]])),
            np.arange(4.0).reshape(2, 2) + 1,
        ),
        (lambda x: sum(x @ np.arange(6.0).reshape(3, 2)), np.arange(6.0).reshape(2, 3)),
        (
            lambda x: sum(x @ np.array([[1.0, 2.0], [3.0, 4.0]])),
            np.array([0.4, -0.7]),
        ),
        (
            lambda x: sum(np.array([0.3, -0.2, 0.5]) @ x),
            np.array([0.1, 0.6, -0.4]),
        ),
        (
            lambda x: sum(x @ np.array([0.3, -0.2, 0.5])),
            np.array([[0.1, 0.6, -0.4], [0.5, 0.2, -0.1]]),
        ),
        (lambda x: sum(relu(x)), np.array([-0.8, 0.4, 1.2])),
        (lambda x: sum(tanh(x)), np.array([-0.8, 0.4, 1.2])),
        (lambda x: sum(sigmoid(x)), np.array([-40.0, 0.4, 40.0])),
        (lambda x: sum(gelu(x)), np.array([-0.8, 0.4, 1.2])),
        (lambda x: sum(softmax(x) * np.array([0.2, -0.4, 0.7])), np.array([0.3, -0.5, 0.8])),
        (
            lambda x: sum(sum(x, axis=(0, 2))),
            np.arange(24.0).reshape(2, 3, 4) / 10,
        ),
    ],
)
def test_operations_match_finite_difference(fn, values) -> None:
    assert check_gradient(fn, values) < 1e-5


def test_batched_matmul_with_broadcasted_operands_matches_finite_difference() -> None:
    left = np.arange(12.0).reshape(2, 2, 3) / 10
    right = np.arange(12.0).reshape(1, 3, 4) / 7
    assert check_gradient(lambda x: sum(x @ right), left) < 1e-5


def test_matmul_accumulates_gradients_for_both_operands() -> None:
    left_values = np.arange(6.0).reshape(2, 3) / 5
    right_values = np.arange(6.0).reshape(3, 2) / 4
    left = Tensor(left_values, requires_grad=True)
    right = Tensor(right_values, requires_grad=True)

    sum(left @ right).backward()

    np.testing.assert_allclose(left.grad, np.ones((2, 2)) @ right_values.T)
    np.testing.assert_allclose(right.grad, left_values.T @ np.ones((2, 2)))


def test_vector_dot_product_matches_finite_difference() -> None:
    right = np.array([0.3, -0.2, 0.5])
    values = np.array([0.1, 0.6, -0.4])
    assert check_gradient(lambda x: x @ right, values) < 1e-5


def test_softmax_cross_entropy_is_stable_for_extreme_logits() -> None:
    logits = Tensor(
        np.array([[10000.0, 9999.0, -10000.0], [-10000.0, 10000.0, 9999.0]]),
        requires_grad=True,
    )
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    loss = softmax_cross_entropy(logits, targets)
    assert np.isfinite(loss.data).all()
    loss.backward()
    assert logits.grad is not None
    assert np.isfinite(logits.grad).all()
    np.testing.assert_allclose(logits.grad.sum(axis=-1), 0.0, atol=1e-12)


def test_softmax_cross_entropy_gradient_matches_finite_difference() -> None:
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = np.array([[0.2, -0.3, 0.8], [-0.4, 0.7, 0.1]])
    assert check_gradient(lambda x: softmax_cross_entropy(x, targets), values) < 1e-5


def test_masked_weights_have_zero_effective_value_and_gradient() -> None:
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [0, 1]])
    weights = Tensor(values, requires_grad=True)
    loss = sum(apply_mask(weights, mask))

    np.testing.assert_array_equal(apply_mask(weights, mask).data, values * mask)
    loss.backward()
    np.testing.assert_array_equal(weights.grad, mask)

    assert check_gradient(lambda x: sum(apply_mask(x, mask)), values) < 1e-5


def test_mask_can_change_between_forward_passes_and_allow_revival() -> None:
    weights = Tensor([1.5, -2.0], requires_grad=True)
    pruned_mask = np.array([1, 0])

    pruned_output = apply_mask(weights, pruned_mask)
    np.testing.assert_array_equal(pruned_output.data, [1.5, 0.0])
    sum(pruned_output).backward()
    np.testing.assert_array_equal(weights.grad, [1.0, 0.0])

    weights.zero_grad()
    revived_mask = np.array([1, 1])
    revived_output = apply_mask(weights, revived_mask)
    np.testing.assert_array_equal(revived_output.data, [1.5, -2.0])
    sum(revived_output).backward()
    np.testing.assert_array_equal(weights.grad, [1.0, 1.0])


def test_gradient_checker_reports_invalid_inputs_and_bad_gradients() -> None:
    def incorrect_gradient(x: Tensor) -> Tensor:
        return Tensor(
            np.sum(x.data**2),
            requires_grad=True,
            parents=(x,),
            op="incorrect_test_gradient",
            backward=lambda grad: (np.zeros_like(x.data),),
        )

    with pytest.raises(AssertionError, match="relative error"):
        check_gradient(incorrect_gradient, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="positive"):
        check_gradient(lambda x: sum(x), np.array([1.0]), eps=0)

    with pytest.raises(ValueError, match="scalar"):
        check_gradient(lambda x: x, np.array([1.0, 2.0]))


def test_backward_requires_explicit_gradient_for_non_scalar_output() -> None:
    output = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="scalar output"):
        output.backward()
    output.backward(np.array([1.0, 3.0]))
    np.testing.assert_array_equal(output.grad, [1.0, 3.0])


def test_mask_must_be_binary_and_broadcastable() -> None:
    weights = Tensor(np.ones((2, 3)), requires_grad=True)
    with pytest.raises(ValueError, match="zeros and ones"):
        apply_mask(weights, np.array([[1.0, 0.5, 1.0]]))
    with pytest.raises(ValueError, match="broadcast"):
        apply_mask(weights, np.ones((4,)))
