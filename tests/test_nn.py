import math
import sys
from pathlib import Path

# Find the project root (the parent of the tests directory)
project_root = Path(__file__).resolve().parent.parent

# Add the project root to Python's import search path
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from engine import Tensor, check_gradient, sum
from nn import Adam, Linear, Parameter, ReLU, Sequential, Sigmoid, Tanh
from nn.init import he_normal, xavier_uniform


def test_he_initialization_uses_relu_fan_in_variance() -> None:
    weights = he_normal((200, 100), np.random.default_rng(12))

    assert abs(weights.mean()) < 0.01
    assert np.var(weights) == pytest.approx(2 / 200, rel=0.06)


def test_xavier_initialization_uses_glorot_bound_and_is_reproducible() -> None:
    first = xavier_uniform((80, 40), np.random.default_rng(12))
    second = xavier_uniform((80, 40), np.random.default_rng(12))
    bound = math.sqrt(6 / (80 + 40))

    np.testing.assert_array_equal(first, second)
    assert np.max(first) <= bound
    assert np.min(first) >= -bound


@pytest.mark.parametrize(
    "initializer, shape",
    [
        (he_normal, (0, 2)),
        (xavier_uniform, (2, -1)),
        (he_normal, (2,)),
    ],
)
def test_initializers_reject_invalid_shapes(initializer, shape) -> None:
    with pytest.raises(ValueError, match="two positive integers"):
        initializer(shape)


def test_linear_initializes_bias_to_zero_and_collects_parameters() -> None:
    layer = Linear(3, 2, rng=np.random.default_rng(4))

    assert layer.W.shape == (3, 2)
    assert layer.b.shape == (2,)
    np.testing.assert_array_equal(layer.b.data, [0.0, 0.0])
    assert layer.parameters() == [layer.W, layer.b]


def test_linear_forward_uses_existing_autodiff_and_accumulates_gradients() -> None:
    layer = Linear(2, 1, activation="linear", rng=np.random.default_rng(3))
    layer.W.data[...] = [[2.0], [-1.0]]
    layer.b.data[...] = [0.5]
    inputs = Tensor([[1.0, 3.0], [-2.0, 1.0]], requires_grad=True)

    output = layer(inputs)
    np.testing.assert_allclose(output.data, [[-0.5], [-4.5]])
    sum(output).backward()

    np.testing.assert_allclose(inputs.grad, [[2.0, -1.0], [2.0, -1.0]])
    np.testing.assert_allclose(layer.W.grad, [[-1.0], [4.0]])
    np.testing.assert_allclose(layer.b.grad, [2.0])


def test_linear_rejects_wrong_input_shape_and_invalid_configuration() -> None:
    layer = Linear(3, 2)
    with pytest.raises(ValueError, match="last dimension"):
        layer(np.ones((4, 2)))
    with pytest.raises(ValueError, match="in_features"):
        Linear(0, 2)
    with pytest.raises(ValueError, match="activation"):
        Linear(2, 3, activation="softmax")


def test_linear_mask_reuses_task1_mask_op_and_reports_optimizer_mask() -> None:
    layer = Linear(2, 1, activation="linear", rng=np.random.default_rng(3))
    layer.W.data[...] = [[2.0], [5.0]]
    layer.mask[...] = [[1.0], [0.0]]

    output = layer(np.array([[3.0, 4.0]]))
    np.testing.assert_allclose(output.data, [[6.0]])
    assert layer.parameter_masks() == {layer.W: layer.mask}
    output.backward()
    np.testing.assert_allclose(layer.W.grad, [[3.0], [0.0]])


def test_sequential_composes_layers_and_collects_parameters_once() -> None:
    first = Linear(2, 4, rng=np.random.default_rng(8))
    last = Linear(4, 3, activation="linear", rng=np.random.default_rng(9))
    model = Sequential(first, ReLU(), last)

    output = model(np.ones((5, 2)))
    assert output.shape == (5, 3)
    assert model.parameters() == [first.W, first.b, last.W, last.b]

    model.zero_grad()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sequential_rejects_non_module_children() -> None:
    with pytest.raises(TypeError, match="Module instances"):
        Sequential(Linear(2, 3), lambda inputs: inputs)


@pytest.mark.parametrize("activation", [ReLU(), Tanh(), Sigmoid()])
def test_activation_modules_reuse_task1_gradients(activation) -> None:
    values = np.array([-0.7, 0.4, 1.1])
    assert check_gradient(lambda x: sum(activation(x)), values) < 1e-5


def test_adam_first_step_matches_bias_corrected_adamw_equation() -> None:
    parameter = Parameter([1.0])
    parameter.grad = np.array([2.0])
    optimizer = Adam([parameter], lr=0.1, weight_decay=0.2)

    optimizer.step()

    expected_update = 2.0 / (abs(2.0) + optimizer.eps)
    np.testing.assert_allclose(parameter.data, [1.0 - 0.02 - 0.1 * expected_update])
    np.testing.assert_allclose(optimizer.m[0], [0.2])
    np.testing.assert_allclose(optimizer.v[0], [0.004])
    assert optimizer.t == 1


def test_adam_updates_moments_and_applies_bias_correction_over_multiple_steps() -> None:
    parameter = Parameter([1.0])
    optimizer = Adam([parameter], lr=0.05, beta1=0.8, beta2=0.9, eps=1e-8)
    parameter.grad = np.array([2.0])
    optimizer.step()
    parameter.grad = np.array([1.0])
    optimizer.step()

    expected_m = 0.8 * 0.4 + 0.2 * 1.0
    expected_v = 0.9 * 0.4 + 0.1 * 1.0
    np.testing.assert_allclose(optimizer.m[0], [expected_m])
    np.testing.assert_allclose(optimizer.v[0], [expected_v])
    assert optimizer.t == 2
    assert parameter.data[0] < 1.0


def test_adam_zero_grad_clears_parameters_before_a_new_batch() -> None:
    parameter = Parameter([2.0])
    optimizer = Adam([parameter])
    parameter.grad = np.array([1.0])

    optimizer.zero_grad()

    assert parameter.grad is None


def test_adam_skips_parameters_without_gradients() -> None:
    parameter = Parameter([2.0])
    optimizer = Adam([parameter])
    optimizer.step()

    np.testing.assert_array_equal(parameter.data, [2.0])
    np.testing.assert_array_equal(optimizer.m[0], [0.0])
    np.testing.assert_array_equal(optimizer.v[0], [0.0])


def test_adam_mask_clears_moments_and_revived_weights_start_fresh() -> None:
    parameter = Parameter([1.0, 1.0])
    optimizer = Adam([parameter], lr=0.1)
    parameter.grad = np.array([1.0, 1.0])
    optimizer.step()
    old_active_moment = optimizer.m[0][0]

    parameter.grad = np.array([2.0, 1000.0])
    optimizer.step(masks={0: np.array([1.0, 0.0])})
    np.testing.assert_array_equal(parameter.data[1], 0.0)
    np.testing.assert_array_equal(optimizer.m[0][1], 0.0)
    np.testing.assert_array_equal(optimizer.v[0][1], 0.0)

    parameter.grad = np.array([1.0, 3.0])
    optimizer.step(masks={0: np.array([1.0, 1.0])})
    np.testing.assert_allclose(optimizer.m[0][1], 0.3)
    assert optimizer.m[0][0] != old_active_moment
    assert parameter.data[1] != 0.0


def test_masked_linear_gradient_and_revived_adam_age_are_fresh() -> None:
    layer = Linear(1, 2, activation="linear", rng=np.random.default_rng(3))
    layer.W.data[...] = [[2.0, 3.0]]
    layer.b.data[...] = [0.0, 0.0]
    layer.mask[...] = [[1.0, 0.0]]
    optimizer = Adam([layer.W], lr=0.1, weight_decay=0.0)

    output = layer(np.array([[4.0]]))
    np.testing.assert_allclose(output.data, [[8.0, 0.0]])
    sum(output).backward()
    np.testing.assert_allclose(layer.W.grad, [[4.0, 0.0]])
    optimizer.step(masks={0: layer.mask})
    np.testing.assert_array_equal(layer.W.data[0, 1], 0.0)
    np.testing.assert_array_equal(optimizer.m[0][0, 1], 0.0)
    np.testing.assert_array_equal(optimizer.v[0][0, 1], 0.0)
    np.testing.assert_array_equal(optimizer.step_counts[0][0, 1], 0)

    for _ in range(5):
        layer.W.grad = np.array([[4.0, 1000.0]])
        optimizer.step(masks={0: layer.mask})
        np.testing.assert_array_equal(layer.W.data[0, 1], 0.0)
        np.testing.assert_array_equal(optimizer.m[0][0, 1], 0.0)
        np.testing.assert_array_equal(optimizer.v[0][0, 1], 0.0)
        np.testing.assert_array_equal(optimizer.step_counts[0][0, 1], 0)

    layer.mask[...] = [[1.0, 1.0]]
    layer.W.grad = np.array([[4.0, 2.0]])
    optimizer.step(masks={0: layer.mask})

    fresh_parameter = Parameter([0.0])
    fresh_optimizer = Adam([fresh_parameter], lr=0.1, weight_decay=0.0)
    fresh_parameter.grad = np.array([2.0])
    fresh_optimizer.step()
    np.testing.assert_allclose(layer.W.data[0, 1], fresh_parameter.data[0])
    np.testing.assert_allclose(optimizer.m[0][0, 1], fresh_optimizer.m[0][0])
    np.testing.assert_allclose(optimizer.v[0][0, 1], fresh_optimizer.v[0][0])
    np.testing.assert_array_equal(optimizer.step_counts[0][0, 1], 1)


def test_adam_enforces_mask_even_when_parameter_has_no_gradient() -> None:
    parameter = Parameter([4.0, -3.0])
    optimizer = Adam([parameter])

    optimizer.step(masks={0: np.array([1.0, 0.0])})

    np.testing.assert_array_equal(parameter.data, [4.0, 0.0])
    np.testing.assert_array_equal(optimizer.m[0], [0.0, 0.0])
    np.testing.assert_array_equal(optimizer.v[0], [0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs, exception, message",
    [
        ({"lr": 0}, ValueError, "lr"),
        ({"lr": float("nan")}, ValueError, "lr"),
        ({"beta1": 1}, ValueError, "beta1"),
        ({"beta2": -0.1}, ValueError, "beta2"),
        ({"eps": 0}, ValueError, "eps"),
        ({"weight_decay": -1}, ValueError, "weight_decay"),
    ],
)
def test_adam_rejects_invalid_hyperparameters(kwargs, exception, message) -> None:
    with pytest.raises(exception, match=message):
        Adam([Parameter([1.0])], **kwargs)


def test_adam_rejects_empty_duplicate_and_non_parameter_inputs() -> None:
    parameter = Parameter([1.0])
    with pytest.raises(ValueError, match="at least one"):
        Adam([])
    with pytest.raises(ValueError, match="duplicates"):
        Adam([parameter, parameter])
    with pytest.raises(TypeError, match="Parameter instances"):
        Adam([Tensor([1.0], requires_grad=True)])


def test_adam_validates_gradient_and_mask_shapes_and_values() -> None:
    parameter = Parameter([1.0, 2.0])
    optimizer = Adam([parameter])
    with pytest.raises(ValueError, match="parameter index"):
        optimizer.step(masks={1: np.ones(2)})
    with pytest.raises(ValueError, match="zeros and ones"):
        optimizer.step(masks={0: np.array([1.0, 0.5])})
    with pytest.raises(ValueError, match="broadcast"):
        optimizer.step(masks={0: np.ones(3)})

    parameter.grad = np.ones(3)
    with pytest.raises(ValueError, match="Gradient shape"):
        optimizer.step()
    parameter.grad = np.array([np.nan, 1.0])
    with pytest.raises(FloatingPointError, match="non-finite"):
        optimizer.step()


def test_adam_rejects_invalid_update_without_mutating_state() -> None:
    parameter = Parameter([2.0])
    optimizer = Adam([parameter], lr=0.1)
    parameter.grad = np.array([np.nan])

    with pytest.raises(FloatingPointError):
        optimizer.step(masks={0: np.array([0.0])})

    np.testing.assert_array_equal(parameter.data, [2.0])
    np.testing.assert_array_equal(optimizer.m[0], [0.0])
    np.testing.assert_array_equal(optimizer.v[0], [0.0])
    assert optimizer.t == 0
