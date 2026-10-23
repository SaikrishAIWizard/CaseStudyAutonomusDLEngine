import json

import numpy as np
import pytest

from nn import Linear, ReLU, Sequential
from train.sweep import (
    CSRMatrix,
    SweepConfig,
    benchmark_csr_inference,
    dense_model_forward,
    linear_to_csr,
    run_pareto_sweep,
    sparse_linear_forward,
    sparse_model_forward,
    theoretical_flops,
    write_sweep_results,
)


def make_masked_model() -> tuple[Sequential, Linear]:
    layer = Linear(3, 2, activation="linear", rng=np.random.default_rng(3))
    layer.W.data[...] = [[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]]
    layer.b.data[...] = [0.25, -0.5]
    layer.mask[...] = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    return Sequential(layer), layer


def test_linear_csr_forward_matches_dense_masked_forward() -> None:
    model, layer = make_masked_model()
    inputs = np.arange(12.0).reshape(4, 3) / 3
    csr = linear_to_csr(layer)

    np.testing.assert_allclose(
        sparse_linear_forward(inputs, csr),
        dense_model_forward(model, inputs),
        rtol=1e-12,
        atol=1e-12,
    )
    assert csr.output_features == 2
    assert csr.input_features == 3
    assert csr.nnz == 3
    assert csr.indptr.tolist() == [0, 2, 3]


def test_sparse_model_forward_matches_dense_model_with_activations() -> None:
    first = Linear(3, 4, activation="relu", rng=np.random.default_rng(1))
    last = Linear(4, 2, activation="linear", rng=np.random.default_rng(2))
    first.mask[0, 1] = 0
    last.mask[2, 0] = 0
    model = Sequential(first, ReLU(), last)
    inputs = np.random.default_rng(4).normal(size=(5, 3))

    np.testing.assert_allclose(
        sparse_model_forward(model, inputs),
        dense_model_forward(model, inputs),
        rtol=1e-12,
        atol=1e-12,
    )


def test_flop_accounting_counts_dense_and_effective_sparse_connections() -> None:
    model, _ = make_masked_model()

    assert theoretical_flops(model, batch_size=4) == 2 * 4 * 6
    assert theoretical_flops(model, batch_size=4, sparse=True) == 2 * 4 * 3


def test_flop_accounting_rejects_invalid_batch_size() -> None:
    model, _ = make_masked_model()
    with pytest.raises(ValueError, match="batch_size"):
        theoretical_flops(model, 0)


def test_csr_benchmark_checks_equivalence_and_reports_observed_costs() -> None:
    model, _ = make_masked_model()
    inputs = np.ones((3, 3))

    result = benchmark_csr_inference(model, inputs, repeats=2, warmup=0)

    assert result["batch_size"] == 3
    assert result["dense_flops"] == 2 * 3 * 6
    assert result["sparse_flops"] == 2 * 3 * 3
    assert result["dense_median_seconds"] > 0
    assert result["csr_median_seconds"] > 0
    assert result["dense_to_csr_speed_ratio"] > 0
    assert result["max_absolute_output_error"] < 1e-12


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"targets": ()}, "targets"),
        ({"targets": (0.5, 0.5)}, "unique"),
        ({"targets": (1.1,)}, "targets"),
        ({"targets": ("0.5",)}, "targets"),
        ({"seeds": ()}, "seeds"),
        ({"seeds": (1, 1)}, "unique"),
        ({"pruning_epochs": 1}, "pruning_epochs"),
        ({"cooldown_epochs": -1}, "cooldown_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"weight_decay": -1}, "weight_decay"),
    ],
)
def test_sweep_configuration_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SweepConfig(**kwargs)


def test_small_pareto_sweep_records_paired_raw_data_and_summary(tmp_path) -> None:
    config = SweepConfig(
        targets=(0.0, 0.5, 0.9),
        seeds=(2, 3),
        pruning_epochs=3,
        cooldown_epochs=1,
        batch_size=8,
        train_samples_per_class=12,
        test_samples_per_class=6,
        benchmark_repeats=1,
        benchmark_warmup=0,
    )
    report = run_pareto_sweep(config)

    assert len(report["trials"]) == 2 * 3 * 2
    assert len(report["summary"]) == 3 * 2
    assert len(report["csr_benchmarks"]) == 3
    assert "synthetic" in report["metadata"]["dataset"]
    assert "paired seeds" in report["quantitative_claim"]
    assert all(
        0 <= trial["test_accuracy"] <= 1
        and 0 <= trial["achieved_sparsity"] <= 1
        and trial["sparse_flops_per_sample"] <= trial["dense_flops_per_sample"]
        for trial in report["trials"]
    )
    for target in config.targets:
        for seed in config.seeds:
            pair = [
                row
                for row in report["trials"]
                if row["seed"] == seed and row["target_sparsity"] == target
            ]
            assert {row["method"] for row in pair} == {"magnitude", "taylor"}

    write_sweep_results(report, tmp_path)
    metrics_path = tmp_path / "pareto_metrics.json"
    plot_path = tmp_path / "pareto_curve.svg"
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["trials"] == report["trials"]
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in plot_path.read_text(
        encoding="utf-8"
    )


def test_sparse_forward_rejects_invalid_csr_structure_and_shapes() -> None:
    valid = CSRMatrix(
        data=np.array([1.0]),
        indices=np.array([0]),
        indptr=np.array([0, 1]),
        bias=np.array([0.0]),
        input_features=2,
    )
    with pytest.raises(ValueError, match="shape"):
        sparse_linear_forward(np.ones(2), valid)

    invalid = CSRMatrix(
        data=np.array([1.0]),
        indices=np.array([3]),
        indptr=np.array([0, 1]),
        bias=np.array([0.0]),
        input_features=2,
    )
    with pytest.raises(ValueError, match="CSR structure"):
        sparse_linear_forward(np.ones((1, 2)), invalid)

    empty_indptr = CSRMatrix(
        data=np.array([]),
        indices=np.array([], dtype=np.int64),
        indptr=np.array([], dtype=np.int64),
        bias=np.array([]),
        input_features=2,
    )
    with pytest.raises(ValueError, match="CSR structure"):
        sparse_linear_forward(np.ones((1, 2)), empty_indptr)


def test_csr_benchmark_rejects_empty_input_batch() -> None:
    model, _ = make_masked_model()
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_csr_inference(model, np.empty((0, 3)), repeats=1, warmup=0)


def test_csr_export_requires_valid_linear_mask() -> None:
    _, layer = make_masked_model()
    layer.mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        linear_to_csr(layer)
