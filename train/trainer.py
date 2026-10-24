from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from engine import Tensor, softmax_cross_entropy
from nn import Adam, Linear, ReLU, Sequential
from nn.module import Module, Parameter
from prune import CubicPruningSchedule, TaylorPruner


@dataclass(frozen=True)
class EpochMetrics:
    """Mean training loss and accuracy for one epoch."""

    epoch: int
    loss: float
    accuracy: float
    sparsity: float = 0.0


def clip_grad_norm(
    parameters: tuple[Parameter, ...] | list[Parameter], max_norm: float
) -> float:
    """Clip all supplied gradients together and return their original norm."""
    if (
        not isinstance(max_norm, (int, float))
        or isinstance(max_norm, bool)
        or not math.isfinite(float(max_norm))
        or max_norm <= 0
    ):
        raise ValueError("max_norm must be a finite positive number")

    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    if any(not np.all(np.isfinite(gradient)) for gradient in gradients):
        raise FloatingPointError("Cannot clip non-finite gradients")

    norm = float(np.sqrt(sum(np.sum(np.square(gradient)) for gradient in gradients)))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad *= scale
    return norm


def train_classifier(
    model: Module,
    optimizer: Adam,
    inputs: ArrayLike,
    labels: ArrayLike,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 0,
    max_grad_norm: float = 5.0,
    pruning_schedule: CubicPruningSchedule | None = None,
    pruner: TaylorPruner | None = None,
    pruning_criterion: Literal["taylor", "magnitude"] = "taylor",
) -> list[EpochMetrics]:
    """Train a classifier with shuffled mini-batches and mean cross-entropy."""
    if not isinstance(model, Module):
        raise TypeError("model must be a Module")
    if not isinstance(optimizer, Adam):
        raise TypeError("optimizer must be an Adam optimizer")
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if (
        not isinstance(max_grad_norm, (int, float))
        or isinstance(max_grad_norm, bool)
        or not math.isfinite(float(max_grad_norm))
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be a finite positive number")
    if (pruning_schedule is None) != (pruner is None):
        raise ValueError("pruning_schedule and pruner must be provided together")
    if pruning_schedule is not None and not isinstance(
        pruning_schedule, CubicPruningSchedule
    ):
        raise TypeError("pruning_schedule must be a CubicPruningSchedule")
    if pruner is not None and not isinstance(pruner, TaylorPruner):
        raise TypeError("pruner must be a TaylorPruner")
    if pruning_criterion not in ("taylor", "magnitude"):
        raise ValueError("pruning_criterion must be 'taylor' or 'magnitude'")

    features = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(labels)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("inputs must be a non-empty 2D array")
    if not np.all(np.isfinite(features)):
        raise ValueError("inputs must contain only finite values")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("labels must be a 1D class-index array matching inputs")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")

    model_parameters = model.parameters()
    if not model_parameters:
        raise ValueError("model must contain trainable parameters")
    if {id(parameter) for parameter in model_parameters} != {
        id(parameter) for parameter in optimizer.params
    }:
        raise ValueError("optimizer parameters must match the model parameters")
    model_parameter_ids = {id(parameter) for parameter in model_parameters}
    if pruner is not None and any(
        id(layer.W) not in model_parameter_ids for layer in pruner.layers
    ):
        raise ValueError("pruner layers must belong to the training model")

    rng = np.random.default_rng(seed)
    parameter_indices = {
        id(parameter): index for index, parameter in enumerate(model_parameters)
    }
    optimizer_masks = {
        parameter_indices[id(parameter)]: mask
        for parameter, mask in model.parameter_masks().items()
    }
    history: list[EpochMetrics] = []
    sample_count = features.shape[0]

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        order = rng.permutation(sample_count)

        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            batch_features = features[indices]
            batch_labels = targets[indices]

            optimizer.zero_grad()
            logits = model(Tensor(batch_features))
            if logits.ndim != 2 or logits.shape[0] != len(indices):
                raise ValueError("model output must have shape (batch_size, classes)")
            if logits.shape[1] < 2:
                raise ValueError("model must output at least two class scores")
            if np.any(batch_labels < 0) or np.any(batch_labels >= logits.shape[1]):
                raise ValueError("labels contain a class index outside model outputs")

            one_hot = np.eye(logits.shape[1], dtype=np.float64)[batch_labels]
            loss = softmax_cross_entropy(logits, one_hot)
            loss_value = float(loss.data.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("Training produced a non-finite loss")

            loss.backward()
            clip_grad_norm(model_parameters, max_grad_norm)
            optimizer.step(masks=optimizer_masks or None)

            epoch_loss += loss_value * len(indices)
            correct += int(
                np.sum(np.argmax(logits.data, axis=1) == batch_labels)
            )

        if (
            pruner is not None
            and pruning_schedule is not None
            and pruning_schedule.should_update(epoch)
        ):
            if pruning_criterion == "taylor":
                calibration_indices = order[: min(batch_size, sample_count)]
                optimizer.zero_grad()
                calibration_logits = model(Tensor(features[calibration_indices]))
                calibration_targets = np.eye(
                    calibration_logits.shape[1], dtype=np.float64
                )[targets[calibration_indices]]
                calibration_loss = softmax_cross_entropy(
                    calibration_logits, calibration_targets
                )
                if not math.isfinite(float(calibration_loss.data.item())):
                    raise FloatingPointError(
                        "Pruning calibration produced a non-finite loss"
                    )
                calibration_loss.backward()

            pruner.step(
                pruning_schedule.sparsity_at(epoch),
                optimizer,
                criterion=pruning_criterion,
            )

        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=epoch_loss / sample_count,
                accuracy=correct / sample_count,
                sparsity=pruner.sparsity if pruner is not None else 0.0,
            )
        )

    return history


def make_demo_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic three-class dataset without extra dependencies."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    features = np.concatenate(
        [center + rng.normal(0.0, 0.55, size=(120, 2)) for center in centers]
    )
    labels = np.repeat(np.arange(len(centers)), 120)
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Task 2 NumPy MLP demo.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    features, labels = make_demo_data(args.seed)
    rng = np.random.default_rng(args.seed)
    model = Sequential(
        Linear(2, 16, activation="relu", rng=rng),
        ReLU(),
        Linear(16, 16, activation="relu", rng=rng),
        ReLU(),
        Linear(16, 3, activation="linear", rng=rng),
    )
    optimizer = Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    history = train_classifier(
        model,
        optimizer,
        features,
        labels,
        epochs=args.epochs,
        batch_size=32,
        seed=args.seed,
    )

    for metrics in history:
        print(
            f"epoch={metrics.epoch:02d} "
            f"loss={metrics.loss:.4f} accuracy={metrics.accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
