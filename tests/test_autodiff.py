
import numpy as np
import sys
from pathlib import Path

# Find the project root (the parent of the tests directory)
project_root = Path(__file__).resolve().parent.parent

# Add the project root to Python's import search path
sys.path.insert(0, str(project_root))

from engine import (
    Tensor,
    sum,
    mean,
    relu,
    check_gradient,
    apply_mask,
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


if __name__ == "__main__":
    test_basic_gradient()
    test_gradient_accumulation()
    test_relu_gradient()
    test_gradient_checker()
    test_masked_weights()

    print("\nAll tests passed!")
