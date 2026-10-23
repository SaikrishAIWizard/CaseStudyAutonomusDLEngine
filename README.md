# Aqua Platform Challenge

NumPy implementation of the challenge's first four parts: reverse-mode
automatic differentiation, MLP training, gradual Taylor pruning, and a
Taylor-versus-magnitude sparsity sweep with CSR inference evaluation.

## Install

Install [uv](https://docs.astral.sh/uv/) if it is not already installed, then
from the repository root run:

```powershell
uv sync --extra test
```

The project requires Python 3.10 or newer. NumPy is the runtime dependency;
pytest is included in the `test` extra. `uv sync` creates and manages `.venv`.
Without uv, create a virtual environment and install `numpy` and `pytest` with
pip.

To run notebooks with the project environment as the Jupyter kernel, install
the optional notebook dependency too:

```powershell
uv sync --extra test --extra notebook
```

## Reproduce each part

Run these commands from the repository root:

| Goal | Command |
| --- | --- |
| Part 1 gradient-checking and autodiff tests | `uv run python -m pytest -q tests/test_autodiff.py` |
| Part 2 MLP training demo | `uv run python -m train.trainer` |
| Part 3 gradual self-pruning demo | `uv run python -m train.pruning_demo` |
| Part 4 Pareto sweep | `uv run python -m train.sweep` |

Part 1's test file also contains
`test_masked_weights_have_zero_effective_value_and_gradient`, which checks
masked-weight forward and gradient correctness. Run all regression tests with:

```powershell
uv run python -m pytest -q
```

Part 4 writes the raw trial metrics and summary to
[`results/pareto_metrics.json`](results/pareto_metrics.json) and the plotted
Pareto curve to [`results/pareto_curve.svg`](results/pareto_curve.svg). The
committed run uses a deterministic synthetic three-class dataset because this
project does not include MNIST; its accuracy values are not MNIST or production
claims. CSR timing is reported as measured and is not assumed to be faster than
dense NumPy.

## Repository layout

```text
engine/       Tensor and reverse-mode autodiff operations/checker
nn/           NumPy layers, parameters, initializers, and AdamW
prune/        Taylor importance, cubic schedule, and global pruner
train/        Training demos and Part 4 sweep
tests/        Gradient, masked-weight, training, pruning, and sweep tests
notebooks/    Interactive task notebooks
results/      Part 4 raw results and Pareto plot
DESIGN.md     Derivations and architecture/design discussion
```

## Tests

The Part 1 suite checks arithmetic, reductions, matrix multiplication,
activations, cross-entropy, and gradients against finite differences. The
masked-weight test verifies that a zero mask removes the weight from the
forward result and gives its stored weight a zero gradient. Other tests cover
the neural-network layers and optimizer, pruning schedules and masks, and the
Part 4 comparison, CSR inference, FLOP counts, validation, and output files.

For implementation rationale, masked-gradient derivation, engine bottlenecks,
and production-serving considerations, see [`DESIGN.md`](DESIGN.md).
