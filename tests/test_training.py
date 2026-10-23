import numpy as np
import pytest

import sys
from pathlib import Path

# Find the project root (the parent of the tests directory)
project_root = Path(__file__).resolve().parent.parent

# Add the project root to Python's import search path
sys.path.insert(0, str(project_root))

from nn import Adam, Linear, ReLU, Sequential
from train.trainer import clip_grad_norm, train_classifier


def make_model(seed: int = 1) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(2, 12, activation="relu", rng=rng),
        ReLU(),
        Linear(12, 3, activation="linear", rng=rng),
    )


def make_data(seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    features = np.concatenate(
        [center + rng.normal(0.0, 0.5, size=(60, 2)) for center in centers]
    )
    labels = np.repeat(np.arange(3), 60)
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def test_training_loop_learns_and_returns_loss_accuracy_per_epoch() -> None:
    features, labels = make_data()
    model = make_model()
    optimizer = Adam(model.parameters(), lr=0.03)

    history = train_classifier(
        model, optimizer, features, labels, epochs=18, batch_size=32, seed=5
    )

    assert len(history) == 18
    assert [entry.epoch for entry in history] == list(range(1, 19))
    assert all(np.isfinite([entry.loss, entry.accuracy]).all() for entry in history)
    assert history[-1].loss < history[0].loss
    assert history[-1].accuracy > 0.95
    assert optimizer.t == 18 * 6


def test_training_is_reproducible_with_fixed_model_and_shuffle_seeds() -> None:
    features, labels = make_data()
    model_a, model_b = make_model(4), make_model(4)
    history_a = train_classifier(
        model_a,
        Adam(model_a.parameters(), lr=0.02),
        features,
        labels,
        epochs=4,
        batch_size=25,
        seed=11,
    )
    history_b = train_classifier(
        model_b,
        Adam(model_b.parameters(), lr=0.02),
        features,
        labels,
        epochs=4,
        batch_size=25,
        seed=11,
    )

    np.testing.assert_allclose(
        [[item.loss, item.accuracy] for item in history_a],
        [[item.loss, item.accuracy] for item in history_b],
    )
    for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
        np.testing.assert_array_equal(parameter_a.data, parameter_b.data)


def test_last_incomplete_batch_is_included_in_epoch_metrics() -> None:
    features, labels = make_data()
    model = make_model()
    optimizer = Adam(model.parameters(), lr=0.01)

    history = train_classifier(
        model, optimizer, features[:65], labels[:65], epochs=1, batch_size=32
    )

    assert len(history) == 1
    assert optimizer.t == 3


def test_training_passes_linear_masks_to_optimizer() -> None:
    features, labels = make_data()
    model = make_model()
    first_linear = model.modules[0]
    first_linear.mask[0, 0] = 0
    optimizer = Adam(model.parameters(), lr=0.02)

    train_classifier(model, optimizer, features, labels, epochs=2, batch_size=30)

    np.testing.assert_array_equal(first_linear.W.data[0, 0], 0.0)
    np.testing.assert_array_equal(optimizer.m[0][0, 0], 0.0)
    np.testing.assert_array_equal(optimizer.v[0][0, 0], 0.0)


def test_gradient_clipping_limits_global_norm() -> None:
    model = make_model()
    parameters = model.parameters()
    parameters[0].grad = np.full(parameters[0].shape, 3.0)
    parameters[1].grad = np.full(parameters[1].shape, 4.0)

    original_norm = clip_grad_norm(parameters, max_norm=2.0)
    clipped_norm = np.sqrt(
        sum(np.sum(parameter.grad**2) for parameter in parameters if parameter.grad is not None)
    )

    assert original_norm > 2.0
    assert clipped_norm == pytest.approx(2.0, rel=1e-10)


@pytest.mark.parametrize(
    "features, labels, message",
    [
        (np.ones(5), np.zeros(5, dtype=int), "2D"),
        (np.ones((5, 2)), np.zeros(4, dtype=int), "matching"),
        (np.ones((5, 2)), np.zeros(5), "integer"),
        (np.full((5, 2), np.nan), np.zeros(5, dtype=int), "finite"),
    ],
)
def test_training_rejects_invalid_data(features, labels, message) -> None:
    model = make_model()
    with pytest.raises(ValueError, match=message):
        train_classifier(model, Adam(model.parameters()), features, labels, epochs=1)


def test_training_rejects_labels_outside_model_classes() -> None:
    model = make_model()
    features = np.ones((4, 2))
    labels = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError, match="outside model outputs"):
        train_classifier(model, Adam(model.parameters()), features, labels, epochs=1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": 1.5}, "seed"),
        ({"max_grad_norm": 0}, "max_grad_norm"),
        ({"max_grad_norm": "bad"}, "max_grad_norm"),
    ],
)
def test_training_rejects_invalid_loop_settings(kwargs, message) -> None:
    features, labels = make_data()
    model = make_model()
    with pytest.raises((TypeError, ValueError), match=message):
        train_classifier(
            model, Adam(model.parameters()), features, labels, **kwargs
        )


def test_training_requires_optimizer_to_cover_model_parameters() -> None:
    features, labels = make_data()
    model = make_model()
    with pytest.raises(ValueError, match="match the model"):
        train_classifier(model, Adam([model.parameters()[0]]), features, labels)


def test_gradient_clipping_rejects_non_finite_gradients() -> None:
    model = make_model()
    parameter = model.parameters()[0]
    parameter.grad = np.full(parameter.shape, np.inf)
    with pytest.raises(FloatingPointError, match="non-finite"):
        clip_grad_norm(model.parameters(), 1.0)
