"""Run a deterministic Task 3 training demo with gradual Taylor pruning."""

import argparse

import numpy as np

from nn import Adam, Linear, ReLU, Sequential
from prune import CubicPruningSchedule, TaylorPruner
from train.trainer import make_demo_data, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLP with a cubic Taylor-pruning schedule."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-sparsity", type=float, default=0.9)
    args = parser.parse_args()
    if args.epochs < 6:
        parser.error("--epochs must be at least 6 to include the pruning warm-up")

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
    pruner = TaylorPruner(model)
    schedule = CubicPruningSchedule(
        final_sparsity=args.target_sparsity,
        begin_step=5,
        end_step=min(20, args.epochs),
        update_frequency=1,
    )
    history = train_classifier(
        model,
        optimizer,
        features,
        labels,
        epochs=args.epochs,
        batch_size=32,
        seed=args.seed,
        pruning_schedule=schedule,
        pruner=pruner,
    )

    for metrics in history:
        print(
            f"epoch={metrics.epoch:02d} "
            f"loss={metrics.loss:.4f} accuracy={metrics.accuracy:.3f} "
            f"sparsity={metrics.sparsity:.3f}"
        )


if __name__ == "__main__":
    main()
