from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from numpy.typing import ArrayLike

from engine import Tensor
from nn import Adam, Linear, ReLU, Sequential, Sigmoid, Tanh
from prune import CubicPruningSchedule, TaylorPruner
from train.trainer import train_classifier


DEFAULT_TARGETS = (0.0, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98)
METHODS = ("magnitude", "taylor")


@dataclass(frozen=True)
class SweepConfig:
    """Reproducible configuration for a multi-seed sparsity sweep."""

    targets: tuple[float, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pruning_epochs: int = 16
    cooldown_epochs: int = 5
    batch_size: int = 32
    train_samples_per_class: int = 80
    test_samples_per_class: int = 40
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    benchmark_repeats: int = 9
    benchmark_warmup: int = 2

    def __post_init__(self) -> None:
        if not self.targets or any(
            not isinstance(target, (int, float))
            or isinstance(target, bool)
            or not math.isfinite(float(target))
            or not 0.0 <= target <= 1.0
            for target in self.targets
        ):
            raise ValueError("targets must be numeric sparsity values in [0, 1]")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be unique")
        if not self.seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds
        ):
            raise ValueError("seeds must be a non-empty tuple of integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        for name, value, minimum in (
            ("pruning_epochs", self.pruning_epochs, 2),
            ("cooldown_epochs", self.cooldown_epochs, 0),
            ("batch_size", self.batch_size, 1),
            ("train_samples_per_class", self.train_samples_per_class, 2),
            ("test_samples_per_class", self.test_samples_per_class, 1),
            ("benchmark_repeats", self.benchmark_repeats, 1),
            ("benchmark_warmup", self.benchmark_warmup, 0),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


@dataclass(frozen=True)
class CSRMatrix:
    """Compressed sparse rows for one Linear layer's output-by-input matrix."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    bias: np.ndarray
    input_features: int

    @property
    def output_features(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.data)


def linear_to_csr(layer: Linear) -> CSRMatrix:
    """Convert active non-zero Linear weights to output-row CSR arrays."""
    if not isinstance(layer, Linear):
        raise TypeError("layer must be a Linear module")
    if layer.mask.shape != layer.W.shape or not np.all(np.isin(layer.mask, (0, 1))):
        raise ValueError("Linear layer must have a valid binary weight mask")

    values: list[float] = []
    input_indices: list[int] = []
    indptr = [0]
    for output_index in range(layer.out_features):
        column = layer.W.data[:, output_index]
        active_inputs = np.flatnonzero((layer.mask[:, output_index] == 1) & (column != 0))
        input_indices.extend(int(index) for index in active_inputs)
        values.extend(float(column[index]) for index in active_inputs)
        indptr.append(len(values))

    return CSRMatrix(
        data=np.asarray(values, dtype=np.float64),
        indices=np.asarray(input_indices, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        bias=layer.b.data.copy(),
        input_features=layer.in_features,
    )


def sparse_linear_forward(inputs: ArrayLike, weights: CSRMatrix) -> np.ndarray:
    """Compute a batch of affine outputs by visiting only stored CSR weights."""
    if not isinstance(weights, CSRMatrix):
        raise TypeError("weights must be a CSRMatrix")
    if (
        not isinstance(weights.input_features, int)
        or isinstance(weights.input_features, bool)
        or weights.input_features <= 0
        or weights.indptr.ndim != 1
        or weights.indices.ndim != 1
        or weights.data.ndim != 1
        or weights.indptr.size == 0
        or not np.issubdtype(weights.indptr.dtype, np.integer)
        or not np.issubdtype(weights.indices.dtype, np.integer)
        or not np.issubdtype(weights.data.dtype, np.number)
        or weights.bias.ndim != 1
        or not np.issubdtype(weights.bias.dtype, np.number)
        or len(weights.indices) != len(weights.data)
        or weights.indptr[0] != 0
        or weights.indptr[-1] != len(weights.data)
        or np.any(np.diff(weights.indptr) < 0)
        or np.any(weights.indices < 0)
        or np.any(weights.indices >= weights.input_features)
        or not np.all(np.isfinite(weights.data))
        or not np.all(np.isfinite(weights.bias))
    ):
        raise ValueError("Invalid CSR structure")
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != weights.input_features:
        raise ValueError(
            f"inputs must have shape (batch, {weights.input_features})"
        )
    if weights.bias.shape != (weights.output_features,):
        raise ValueError("CSR bias shape does not match output feature count")
    if len(weights.indptr) != weights.output_features + 1:
        raise ValueError("Invalid CSR structure")

    output = np.zeros((values.shape[0], weights.output_features), dtype=np.float64)
    for output_index in range(weights.output_features):
        start, end = weights.indptr[output_index : output_index + 2]
        if start != end:
            output[:, output_index] = (
                values[:, weights.indices[start:end]] @ weights.data[start:end]
            )
    output += weights.bias
    return output


def sparse_model_forward(model: Sequential, inputs: ArrayLike) -> np.ndarray:
    """Run a Sequential Linear/activation model through the explicit CSR path."""
    return _forward_csr_layers(compile_csr_model(model), inputs)


def compile_csr_model(
    model: Sequential,
) -> tuple[CSRMatrix | ReLU | Tanh | Sigmoid, ...]:
    """Build reusable CSR weights for a supported Sequential inference model."""
    if not isinstance(model, Sequential):
        raise TypeError("model must be a Sequential module")
    compiled: list[CSRMatrix | ReLU | Tanh | Sigmoid] = []
    for module in model.modules:
        if isinstance(module, Linear):
            compiled.append(linear_to_csr(module))
        elif isinstance(module, (ReLU, Tanh, Sigmoid)):
            compiled.append(module)
        else:
            raise TypeError(
                f"CSR inference does not support {type(module).__name__}"
            )
    return tuple(compiled)


def _forward_csr_layers(
    layers: tuple[CSRMatrix | ReLU | Tanh | Sigmoid, ...], inputs: ArrayLike
) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("inputs must be a 2D batch")
    for module in layers:
        if isinstance(module, CSRMatrix):
            values = sparse_linear_forward(values, module)
        elif isinstance(module, ReLU):
            values = np.maximum(values, 0.0)
        elif isinstance(module, Tanh):
            values = np.tanh(values)
        elif isinstance(module, Sigmoid):
            positive = values >= 0
            output = np.empty_like(values)
            output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
            exp_values = np.exp(values[~positive])
            output[~positive] = exp_values / (1.0 + exp_values)
            values = output
    return values


def dense_model_forward(model: Sequential, inputs: ArrayLike) -> np.ndarray:
    """Run equivalent dense NumPy inference, including masked zero weights."""
    if not isinstance(model, Sequential):
        raise TypeError("model must be a Sequential module")
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("inputs must be a 2D batch")
    for module in model.modules:
        if isinstance(module, Linear):
            values = values @ (module.W.data * module.mask) + module.b.data
        elif isinstance(module, ReLU):
            values = np.maximum(values, 0.0)
        elif isinstance(module, Tanh):
            values = np.tanh(values)
        elif isinstance(module, Sigmoid):
            positive = values >= 0
            output = np.empty_like(values)
            output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
            exp_values = np.exp(values[~positive])
            output[~positive] = exp_values / (1.0 + exp_values)
            values = output
        else:
            raise TypeError(
                f"Dense NumPy inference does not support {type(module).__name__}"
            )
    return values


def theoretical_flops(
    model: Sequential, batch_size: int, *, sparse: bool = False
) -> int:
    """Count multiply-add FLOPs, excluding bias and activation operations."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(model, Sequential):
        raise TypeError("model must be a Sequential module")
    connections = 0
    for module in model.modules:
        if isinstance(module, Linear):
            connections += (
                int(np.count_nonzero((module.mask == 1) & (module.W.data != 0)))
                if sparse
                else module.W.data.size
            )
    return 2 * batch_size * connections


def benchmark_csr_inference(
    model: Sequential,
    inputs: ArrayLike,
    *,
    repeats: int = 9,
    warmup: int = 2,
) -> dict[str, float | int]:
    """Check dense/CSR equivalence and report median measured inference times."""
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    input_values = np.asarray(inputs, dtype=np.float64)
    if input_values.ndim != 2 or input_values.shape[0] == 0:
        raise ValueError("inputs must be a non-empty 2D batch")
    build_start = time.perf_counter()
    compiled_layers = compile_csr_model(model)
    csr_build_seconds = time.perf_counter() - build_start
    dense_output = dense_model_forward(model, inputs)
    sparse_output = _forward_csr_layers(compiled_layers, inputs)
    np.testing.assert_allclose(sparse_output, dense_output, rtol=1e-10, atol=1e-10)

    for _ in range(warmup):
        dense_model_forward(model, inputs)
        _forward_csr_layers(compiled_layers, inputs)

    dense_times: list[float] = []
    sparse_times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        dense_model_forward(model, inputs)
        dense_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        _forward_csr_layers(compiled_layers, inputs)
        sparse_times.append(time.perf_counter() - start)

    dense_median = statistics.median(dense_times)
    sparse_median = statistics.median(sparse_times)
    return {
        "batch_size": int(np.asarray(inputs).shape[0]),
        "dense_flops": theoretical_flops(
            model, int(np.asarray(inputs).shape[0]), sparse=False
        ),
        "sparse_flops": theoretical_flops(
            model, int(np.asarray(inputs).shape[0]), sparse=True
        ),
        "dense_median_seconds": dense_median,
        "csr_median_seconds": sparse_median,
        "csr_build_seconds": csr_build_seconds,
        "dense_to_csr_speed_ratio": dense_median / sparse_median,
        "max_absolute_output_error": float(np.max(np.abs(dense_output - sparse_output))),
    }


def _make_dataset(
    seed: int, train_per_class: int, test_per_class: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])

    def sample(per_class: int) -> tuple[np.ndarray, np.ndarray]:
        features = np.concatenate(
            [center + rng.normal(0.0, 0.55, size=(per_class, 2)) for center in centers]
        )
        labels = np.repeat(np.arange(len(centers)), per_class)
        order = rng.permutation(len(labels))
        return features[order], labels[order]

    train_x, train_y = sample(train_per_class)
    test_x, test_y = sample(test_per_class)
    return train_x, train_y, test_x, test_y


def _make_model(seed: int) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(2, 16, activation="relu", rng=rng),
        ReLU(),
        Linear(16, 3, activation="linear", rng=rng),
    )


def _accuracy(model: Sequential, inputs: np.ndarray, labels: np.ndarray) -> float:
    logits = model(Tensor(inputs)).data
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def run_pareto_sweep(
    config: SweepConfig = SweepConfig(),
) -> dict[str, object]:
    """Train dense, magnitude-pruned, and Taylor-pruned models across seeds."""
    if not isinstance(config, SweepConfig):
        raise TypeError("config must be a SweepConfig")
    trials: list[dict[str, object]] = []
    csr_benchmarks: list[dict[str, object]] = []
    total_epochs = config.pruning_epochs + config.cooldown_epochs

    for seed in config.seeds:
        train_x, train_y, test_x, test_y = _make_dataset(
            seed, config.train_samples_per_class, config.test_samples_per_class
        )
        for target in config.targets:
            for method in METHODS:
                model = _make_model(seed)
                optimizer = Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                pruner = TaylorPruner(model)
                schedule = CubicPruningSchedule(
                    final_sparsity=target,
                    begin_step=1,
                    end_step=config.pruning_epochs,
                )
                train_classifier(
                    model,
                    optimizer,
                    train_x,
                    train_y,
                    epochs=total_epochs,
                    batch_size=config.batch_size,
                    seed=seed,
                    pruning_schedule=schedule,
                    pruner=pruner,
                    max_grad_norm=5.0,
                    pruning_criterion=method,
                )

                accuracy = _accuracy(model, test_x, test_y)
                active = sum(
                    int(np.count_nonzero(layer.mask)) for layer in pruner.layers
                )
                total = sum(layer.mask.size for layer in pruner.layers)
                achieved_sparsity = pruner.sparsity
                dense_flops = theoretical_flops(
                    model, batch_size=1, sparse=False
                )
                sparse_flops = theoretical_flops(
                    model, batch_size=1, sparse=True
                )
                trials.append(
                    {
                        "seed": seed,
                        "method": method,
                        "target_sparsity": target,
                        "achieved_sparsity": achieved_sparsity,
                        "test_accuracy": accuracy,
                        "active_connections": active,
                        "total_connections": total,
                        "dense_flops_per_sample": dense_flops,
                        "sparse_flops_per_sample": sparse_flops,
                    }
                )

                if seed == config.seeds[0] and method == "taylor":
                    benchmark = benchmark_csr_inference(
                        model,
                        test_x[: min(config.batch_size, len(test_x))],
                        repeats=config.benchmark_repeats,
                        warmup=config.benchmark_warmup,
                    )
                    csr_benchmarks.append(
                        {
                            "seed": seed,
                            "target_sparsity": target,
                            "achieved_sparsity": achieved_sparsity,
                            **benchmark,
                        }
                    )

    summary = _summarize_trials(trials, config.targets)
    return {
        "metadata": {
            "dataset": "deterministic three-class Gaussian synthetic classification",
            "test_accuracy": "measured on held-out synthetic samples",
            "epochs": total_epochs,
            "pruning_epochs": config.pruning_epochs,
            "cooldown_epochs": config.cooldown_epochs,
            "seeds": list(config.seeds),
            "targets": list(config.targets),
            "model": "2-16-3 MLP with ReLU",
            "methods": list(METHODS),
            "flop_formula": "2 * batch_size * active/nonzero weights; bias and activations excluded",
            "sparse_execution": "custom NumPy CSR output-row kernel; benchmark is a microbenchmark, not a speedup guarantee",
        },
        "trials": trials,
        "summary": summary,
        "csr_benchmarks": csr_benchmarks,
        "quantitative_claim": _make_claim(summary),
    }


def _summarize_trials(
    trials: list[dict[str, object]], targets: tuple[float, ...]
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for target in targets:
        for method in METHODS:
            group = [
                trial
                for trial in trials
                if trial["target_sparsity"] == target and trial["method"] == method
            ]
            accuracies = [float(trial["test_accuracy"]) for trial in group]
            sparsities = [float(trial["achieved_sparsity"]) for trial in group]
            summary.append(
                {
                    "target_sparsity": target,
                    "method": method,
                    "mean_test_accuracy": statistics.mean(accuracies),
                    "std_test_accuracy": (
                        statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                    ),
                    "mean_achieved_sparsity": statistics.mean(sparsities),
                    "seeds": len(group),
                }
            )
    return summary


def _make_claim(summary: list[dict[str, object]]) -> str:
    taylor_at_90 = next(
        (
            row
            for row in summary
            if row["method"] == "taylor" and row["target_sparsity"] == 0.9
        ),
        None,
    )
    magnitude_at_90 = next(
        (
            row
            for row in summary
            if row["method"] == "magnitude" and row["target_sparsity"] == 0.9
        ),
        None,
    )
    if taylor_at_90 is None or magnitude_at_90 is None:
        return "No 90% target was included; use the raw sweep results for a claim."
    difference = (
        float(taylor_at_90["mean_test_accuracy"])
        - float(magnitude_at_90["mean_test_accuracy"])
    )
    return (
        f"On the configured held-out synthetic classification benchmark at a "
        f"90% target sparsity, Taylor pruning achieved mean accuracy "
        f"{float(taylor_at_90['mean_test_accuracy']):.4f} "
        f"(SD {float(taylor_at_90['std_test_accuracy']):.4f}) versus magnitude "
        f"pruning {float(magnitude_at_90['mean_test_accuracy']):.4f} "
        f"(SD {float(magnitude_at_90['std_test_accuracy']):.4f}) across "
        f"{int(taylor_at_90['seeds'])} paired seeds; paired difference in "
        f"means: {difference:+.4f}. This is a result for the stated synthetic "
        f"benchmark, not an MNIST or production accuracy claim."
    )


def write_sweep_results(report: dict[str, object], output_dir: str | Path) -> None:
    """Write raw trial data, grouped results, and a dependency-free SVG plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "pareto_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    (output_path / "pareto_curve.svg").write_text(
        _make_svg(report["summary"]), encoding="utf-8"
    )


def _make_svg(summary: object) -> str:
    rows = summary
    if not isinstance(rows, list) or not rows:
        raise ValueError("summary must be a non-empty list")

    width, height = 900, 560
    left, right, top, bottom = 90, 35, 45, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    y_min, y_max = 0.0, 1.0

    def x_coordinate(sparsity: float) -> float:
        return left + sparsity * plot_width

    def y_coordinate(accuracy: float) -> float:
        return top + (y_max - accuracy) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">Held-out accuracy vs. achieved sparsity</text>',
    ]
    for tick in range(0, 101, 20):
        x = x_coordinate(tick / 100)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick}%</text>'
        )
    for tick in range(0, 101, 20):
        accuracy = tick / 100
        y = y_coordinate(accuracy)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{accuracy:.0%}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
            f'<text x="{left + plot_width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">Achieved weight sparsity</text>',
            f'<text x="22" y="{top + plot_height/2:.1f}" transform="rotate(-90 22 {top + plot_height/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="14">Test accuracy</text>',
        ]
    )

    colors = {"magnitude": "#d95f02", "taylor": "#1b9e77"}
    for method in METHODS:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["mean_achieved_sparsity"]),
        )
        points = " ".join(
            f'{x_coordinate(float(row["mean_achieved_sparsity"])):.2f},'
            f'{y_coordinate(float(row["mean_test_accuracy"])):.2f}'
            for row in method_rows
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>'
        )
        for row in method_rows:
            x = x_coordinate(float(row["mean_achieved_sparsity"]))
            accuracy = float(row["mean_test_accuracy"])
            error = float(row["std_test_accuracy"])
            upper = y_coordinate(min(y_max, accuracy + error))
            lower = y_coordinate(max(y_min, accuracy - error))
            y = y_coordinate(accuracy)
            parts.append(
                f'<line x1="{x:.2f}" y1="{upper:.2f}" x2="{x:.2f}" y2="{lower:.2f}" stroke="{colors[method]}"/>'
            )
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[method]}"><title>{method}: accuracy {accuracy:.4f} ± {error:.4f}, sparsity {float(row["mean_achieved_sparsity"]):.4f}</title></circle>'
            )
        legend_y = top + 18 * (1 + METHODS.index(method))
        parts.append(
            f'<line x1="{width-165}" y1="{legend_y}" x2="{width-140}" y2="{legend_y}" stroke="{colors[method]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{width-132}" y="{legend_y+4}" font-family="sans-serif" font-size="12">{method.title()}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Task 4 sparsity and CSR sweep."
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--targets", nargs="+", type=float, default=list(DEFAULT_TARGETS))
    parser.add_argument("--pruning-epochs", type=int, default=16)
    parser.add_argument("--cooldown-epochs", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=9)
    args = parser.parse_args()

    config = SweepConfig(
        targets=tuple(args.targets),
        seeds=tuple(args.seeds),
        pruning_epochs=args.pruning_epochs,
        cooldown_epochs=args.cooldown_epochs,
        benchmark_repeats=args.benchmark_repeats,
    )
    report = run_pareto_sweep(config)
    write_sweep_results(report, args.output)
    print(f"Saved raw trials and summary to {args.output / 'pareto_metrics.json'}")
    print(f"Saved Pareto plot to {args.output / 'pareto_curve.svg'}")
    print(report["quantitative_claim"])
    for row in report["csr_benchmarks"]:
        print(
            "CSR benchmark: "
            f"target={row['target_sparsity']:.0%}, "
            f"achieved={row['achieved_sparsity']:.2%}, "
            f"dense_flops={row['dense_flops']}, "
            f"sparse_flops={row['sparse_flops']}, "
            f"dense_ms={row['dense_median_seconds'] * 1000:.3f}, "
            f"csr_ms={row['csr_median_seconds'] * 1000:.3f}, "
            f"csr_build_ms={row['csr_build_seconds'] * 1000:.3f}, "
            f"speed_ratio={row['dense_to_csr_speed_ratio']:.3f}x"
        )


if __name__ == "__main__":
    main()
