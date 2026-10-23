import numpy as np
import pytest

from nn import Adam, Linear, Sequential
from prune import CubicPruningSchedule, TaylorPruner, taylor_saliency
from train.trainer import train_classifier


def make_layer_model(seed: int = 1) -> tuple[Sequential, Linear]:
    layer = Linear(2, 2, activation="linear", rng=np.random.default_rng(seed))
    return Sequential(layer), layer


def test_taylor_saliency_is_absolute_weight_times_gradient() -> None:
    weights = np.array([[2.0, -3.0], [0.5, 4.0]])
    gradients = np.array([[-0.25, 0.5], [2.0, -0.1]])

    np.testing.assert_allclose(
        taylor_saliency(weights, gradients),
        [[0.5, 1.5], [1.0, 0.4]],
    )


def test_taylor_saliency_rejects_mismatched_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="does not match"):
        taylor_saliency(np.ones((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="Weights"):
        taylor_saliency([[np.inf]], [[1.0]])
    with pytest.raises(ValueError, match="Gradients"):
        taylor_saliency([[1.0]], [[np.nan]])


def test_cubic_schedule_has_warmup_ramp_and_final_cooldown_target() -> None:
    schedule = CubicPruningSchedule(
        initial_sparsity=0.0,
        final_sparsity=0.9,
        begin_step=5,
        end_step=20,
        update_frequency=3,
    )

    assert schedule.sparsity_at(0) == 0.0
    assert schedule.sparsity_at(5) == 0.0
    assert schedule.sparsity_at(12) == pytest.approx(
        0.9 * (1.0 - (8.0 / 15.0) ** 3)
    )
    assert schedule.sparsity_at(20) == 0.9
    assert schedule.sparsity_at(100) == 0.9

    assert not schedule.should_update(4)
    assert schedule.should_update(5)
    assert not schedule.should_update(6)
    assert schedule.should_update(8)
    assert schedule.should_update(20)
    assert not schedule.should_update(21)


@pytest.mark.parametrize(
    "kwargs, exception, message",
    [
        ({"initial_sparsity": -0.1}, ValueError, "initial_sparsity"),
        ({"final_sparsity": 1.1}, ValueError, "final_sparsity"),
        (
            {"initial_sparsity": 0.8, "final_sparsity": 0.2},
            ValueError,
            "greater than or equal",
        ),
        ({"begin_step": 5, "end_step": 5}, ValueError, "end_step"),
        ({"update_frequency": 0}, ValueError, "update_frequency"),
        ({"final_sparsity": float("nan")}, ValueError, "final_sparsity"),
    ],
)
def test_cubic_schedule_rejects_invalid_configuration(
    kwargs, exception, message
) -> None:
    with pytest.raises(exception, match=message):
        CubicPruningSchedule(**kwargs)


def test_cubic_schedule_rejects_non_integer_step_values() -> None:
    schedule = CubicPruningSchedule(begin_step=1, end_step=5)

    with pytest.raises(ValueError, match="step"):
        schedule.sparsity_at(1.5)
    with pytest.raises(ValueError, match="step"):
        schedule.should_update(-1)


def test_global_taylor_pruner_hits_exact_budget_and_enforces_binary_masks() -> None:
    model, layer = make_layer_model()
    layer.W.data[...] = [[1.0, 2.0], [3.0, 4.0]]
    layer.W.grad = np.array([[0.1, 2.0], [1.0, 0.05]])
    optimizer = Adam(model.parameters(), lr=0.01)
    pruner = TaylorPruner(model)

    result = pruner.step(0.5, optimizer)

    assert result.pruned_connections == 2
    assert result.active_connections == 2
    assert result.total_connections == 4
    assert result.achieved_sparsity == 0.5
    assert pruner.sparsity == 0.5
    np.testing.assert_array_equal(layer.mask, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(layer.W.data, [[0.0, 2.0], [3.0, 0.0]])
    np.testing.assert_array_equal(optimizer.m[0], np.zeros_like(layer.W.data))
    np.testing.assert_array_equal(optimizer.v[0], np.zeros_like(layer.W.data))
    np.testing.assert_array_equal(
        optimizer.step_counts[0], np.zeros_like(layer.W.data, dtype=np.int64)
    )


def test_pruner_uses_stable_order_for_equal_saliency_scores() -> None:
    model, layer = make_layer_model()
    layer.W.data.fill(1.0)
    layer.W.grad = np.array([[0.1, 0.1], [0.2, 0.3]])

    result = TaylorPruner(model).step(0.25, Adam(model.parameters()))

    assert result.pruned_connections == 1
    np.testing.assert_array_equal(layer.mask, [[0.0, 1.0], [1.0, 1.0]])


def test_pruner_targets_global_sparsity_across_layers() -> None:
    first = Linear(2, 2, activation="linear", rng=np.random.default_rng(2))
    second = Linear(2, 2, activation="linear", rng=np.random.default_rng(3))
    model = Sequential(first, second)
    first.W.data.fill(1.0)
    second.W.data.fill(1.0)
    first.W.grad = np.full((2, 2), 1.0)
    second.W.grad = np.full((2, 2), 2.0)

    result = TaylorPruner(model).step(0.5, Adam(model.parameters()))

    assert result.total_connections == 8
    assert result.active_connections == 4
    assert result.achieved_sparsity == 0.5
    assert np.count_nonzero(first.mask) == 0
    assert np.count_nonzero(second.mask) == 4


def test_pruner_cannot_reduce_existing_sparsity_without_regrowth() -> None:
    model, layer = make_layer_model()
    layer.W.grad = np.ones(layer.W.shape)
    layer.mask[0, 0] = 0

    with pytest.raises(ValueError, match="does not regrow"):
        TaylorPruner(model).step(0.0, Adam(model.parameters()))


def test_pruner_requires_linear_weights_gradients_and_matching_optimizer() -> None:
    model, layer = make_layer_model()
    optimizer = Adam(model.parameters())
    pruner = TaylorPruner(model)
    with pytest.raises(RuntimeError, match="requires a gradient"):
        pruner.step(0.5, optimizer)

    layer.W.grad = np.ones(layer.W.shape)
    with pytest.raises(ValueError, match="every pruned layer"):
        pruner.step(0.5, Adam([layer.b]))

    with pytest.raises(ValueError, match="between zero and one"):
        pruner.step(1.1, optimizer)


@pytest.mark.parametrize("bad_mask", [np.full((2, 2), 0.5), np.ones((3, 2))])
def test_pruner_rejects_invalid_layer_masks(bad_mask) -> None:
    model, layer = make_layer_model()
    layer.mask = bad_mask

    with pytest.raises(ValueError, match="Mask|masks"):
        TaylorPruner(model)


def test_pruner_clears_optimizer_state_as_soon_as_weights_are_pruned() -> None:
    model, layer = make_layer_model()
    optimizer = Adam(model.parameters(), lr=0.1)
    layer.W.grad = np.ones(layer.W.shape)
    optimizer.step()
    layer.W.grad = np.array([[0.01, 1.0], [1.0, 1.0]])

    TaylorPruner(model).step(0.25, optimizer)

    np.testing.assert_array_equal(layer.mask[0, 0], 0.0)
    np.testing.assert_array_equal(layer.W.data[0, 0], 0.0)
    np.testing.assert_array_equal(optimizer.m[0][0, 0], 0.0)
    np.testing.assert_array_equal(optimizer.v[0][0, 0], 0.0)
    np.testing.assert_array_equal(optimizer.step_counts[0][0, 0], 0)


def test_training_loop_applies_cubic_pruning_and_reports_sparsity() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(12, 2))
    labels = np.arange(12, dtype=np.int64) % 3
    layer = Linear(2, 4, activation="relu", rng=np.random.default_rng(9))
    output = Linear(4, 3, activation="linear", rng=np.random.default_rng(10))
    model = Sequential(layer, output)
    optimizer = Adam(model.parameters(), lr=0.02)

    history = train_classifier(
        model,
        optimizer,
        features,
        labels,
        epochs=3,
        batch_size=6,
        seed=11,
        pruning_schedule=CubicPruningSchedule(
            final_sparsity=0.5, begin_step=1, end_step=3
        ),
        pruner=TaylorPruner(model),
    )

    assert [metrics.sparsity for metrics in history] == [0.0, 0.4, 0.5]
    assert all(np.isfinite([metrics.loss, metrics.accuracy]).all() for metrics in history)
    for current, previous in zip(history[1:], history[:-1]):
        assert current.sparsity >= previous.sparsity
    for pruned_layer in (layer, output):
        np.testing.assert_array_equal(
            pruned_layer.W.data[pruned_layer.mask == 0], 0.0
        )
        parameter_index = next(
            index
            for index, parameter in enumerate(optimizer.params)
            if parameter is pruned_layer.W
        )
        inactive = pruned_layer.mask == 0
        np.testing.assert_array_equal(optimizer.m[parameter_index][inactive], 0.0)
        np.testing.assert_array_equal(optimizer.v[parameter_index][inactive], 0.0)


def test_training_rejects_partial_or_foreign_pruning_configuration() -> None:
    model, _ = make_layer_model()
    features = np.ones((3, 2))
    labels = np.array([0, 1, 0])
    optimizer = Adam(model.parameters())

    with pytest.raises(ValueError, match="provided together"):
        train_classifier(
            model,
            optimizer,
            features,
            labels,
            epochs=1,
            pruner=TaylorPruner(model),
        )

    foreign_model, _ = make_layer_model(seed=4)
    with pytest.raises(ValueError, match="belong to the training model"):
        train_classifier(
            model,
            optimizer,
            features,
            labels,
            epochs=1,
            pruning_schedule=CubicPruningSchedule(end_step=2),
            pruner=TaylorPruner(foreign_model),
        )

    with pytest.raises(TypeError, match="CubicPruningSchedule"):
        train_classifier(
            model,
            optimizer,
            features,
            labels,
            epochs=1,
            pruning_schedule="not a schedule",
            pruner=TaylorPruner(model),
        )
