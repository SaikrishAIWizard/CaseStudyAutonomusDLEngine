
import sys
from pathlib import Path

import numpy as np
import pytest

# Find the project root (the parent of the tests directory)
project_root = Path(__file__).resolve().parent.parent

# Add the project root to Python's import search path
sys.path.insert(0, str(project_root))

from engine import (
    Tensor,
    add,
    div,
    gelu,
    matmul,
    mean,
    mul,
    neg,
    power,
    relu,
    check_gradient,
    apply_mask,
    sigmoid,
    softmax,
    softmax_cross_entropy,
    sub,
    sum,
    tanh,
)


# TEST 1: Check whether backward calculates correct gradients
def test_basic_gradient():
    x = Tensor([2.0, 3.0], requires_grad=True)

    # Loss = mean(x * x) = (2² + 3²) / 2 = 6.5
    loss = mean(x * x)

    loss.backward()

    assert loss.data.item() == 6.5
    np.testing.assert_allclose(x.grad, [2.0, 3.0])

    print("Test 1 passed: Basic gradients")


# TEST 2: Check whether gradients accumulate correctly
def test_gradient_accumulation():
    x = Tensor(3.0, requires_grad=True)

    # Loss = x*x + x
    loss = x * x + x

    loss.backward()

    # dL/dx = 2*x + 1 = 7
    np.testing.assert_allclose(x.grad, 7.0)

    print("Test 2 passed: Gradient accumulation")


# TEST 3: Check whether ReLU blocks negative-value gradients
def test_relu_gradient():
    x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)

    output = relu(x)
    sum(output).backward()

    # ReLU output = [0, 0, 3]
    np.testing.assert_array_equal(output.data, [0.0, 0.0, 3.0])

    # ReLU gradient = [0, 0, 1]
    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0])

    print("Test 3 passed: ReLU gradients")


# TEST 4: Compare autodiff gradients with numerical gradients
def test_gradient_checker():
    def objective(x):
        return mean(x * x)

    values = np.array([2.0, 3.0])

    error = check_gradient(objective, values)

    print("Gradient checking error:", error)

    assert error < 1e-5

    print("Test 4 passed: Gradient checker")


@pytest.mark.parametrize(
    "eps,tol",
    [
        (float("nan"), 1e-5),
        (float("inf"), 1e-5),
        (1e-5, float("nan")),
        (1e-5, float("inf")),
        (1e-5, float("-inf")),
    ],
)
def test_gradient_checker_rejects_non_finite_parameters(eps, tol):
    with pytest.raises(ValueError):
        check_gradient(lambda x: mean(x * x), [1.0, 2.0], eps=eps, tol=tol)


@pytest.mark.parametrize(
    "objective,values",
    [
        (lambda x: sum(add(x, np.array([0.2, -0.3]))), [0.7, 1.2]),
        (lambda x: sum(sub(np.array([0.2, -0.3]), x)), [0.7, 1.2]),
        (lambda x: sum(mul(x, np.array([0.2, -0.3]))), [0.7, 1.2]),
        (lambda x: sum(div(np.array([0.2, -0.3]), x)), [0.7, 1.2]),
        (lambda x: sum(div(x, np.array([0.8, 1.3]))), [0.7, 1.2]),
        (lambda x: sum(neg(x)), [0.7, 1.2]),
        (lambda x: sum(power(x, 3.0)), [0.7, 1.2]),
        (
            lambda x: sum(matmul(x, np.array([[0.2, -0.3], [0.4, 0.1]]))),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
        (
            lambda x: sum(matmul(np.array([[0.2, -0.3], [0.4, 0.1]]), x)),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
        (
            lambda x: sum(matmul(x, np.array([0.2, -0.3]))),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
        (
            lambda x: sum(matmul(np.array([0.2, -0.3]), x)),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
        (
            lambda x: matmul(x, np.array([0.2, -0.3])),
            [0.7, 1.2],
        ),
        (
            lambda x: sum(mean(x, axis=1, keepdims=True)),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
        (
            lambda x: sum(sum(x, axis=(0, 2))),
            np.arange(24.0).reshape(2, 3, 4) / 10 + 0.3,
        ),
        (lambda x: sum(relu(x)), [-0.7, 0.4, 1.1]),
        (lambda x: sum(tanh(x)), [-0.7, 0.4, 1.1]),
        (lambda x: sum(sigmoid(x)), [-0.7, 0.4, 1.1]),
        (lambda x: sum(gelu(x)), [-0.7, 0.4, 1.1]),
        (
            lambda x: sum(
                softmax(x) * np.array([[0.3, -0.2, 0.8], [-0.6, 0.4, 0.1]])
            ),
            [[0.7, 1.2, -0.4], [-0.2, 0.5, 0.9]],
        ),
        (
            lambda x: softmax_cross_entropy(
                x, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            ),
            [[0.7, 1.2, -0.4], [-0.2, 0.5, 0.9]],
        ),
        (
            lambda x: sum(apply_mask(x, np.array([[1.0, 0.0], [0.0, 1.0]]))),
            [[0.7, 1.2], [-0.2, 0.5]],
        ),
    ],
)
def test_implemented_operation_gradients_match_finite_differences(objective, values):
    assert check_gradient(objective, values) < 1e-5


def test_gradient_checker_handles_broadcasted_binary_gradients():
    values = np.array([[0.7, 1.2], [-0.2, 0.5]])
    bias = np.array([[0.3, -0.4]])
    assert check_gradient(lambda x: mean((x + bias) * (x - bias)), values) < 1e-5


# TEST 5: Check whether a mask blocks selected weights
def test_masked_weights():
    weights = Tensor([[2.0, 5.0]], requires_grad=True)

    # Keep the first weight; block the second
    mask = np.array([[1.0, 0.0]])

    effective_weights = apply_mask(weights, mask)
    loss = sum(effective_weights)

    loss.backward()

    np.testing.assert_array_equal(
        effective_weights.data, [[2.0, 0.0]]
    )
    np.testing.assert_array_equal(
        weights.grad, [[1.0, 0.0]]
    )

    print("Test 5 passed: Weight masking")


@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3)])
def test_cross_entropy_rejects_empty_batch(shape):
    logits = np.empty(shape, dtype=np.float64)
    targets = np.empty(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="at least one target row"):
        softmax_cross_entropy(logits, targets)


if __name__ == "__main__":
    test_basic_gradient()
    test_gradient_accumulation()
    test_relu_gradient()
    test_gradient_checker()
    test_masked_weights()

    print("\nAll tests passed!")
