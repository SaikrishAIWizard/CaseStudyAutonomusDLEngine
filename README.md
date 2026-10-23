# Aqua Case Study - Tasks 1 to 4

This repository implements Tasks 1-4 of the Aqua Platform challenge: a
reverse-mode autodiff engine, a NumPy-only MLP with AdamW training,
Taylor-saliency pruning with a gradual cubic schedule, and sparsity/accuracy
and sparse-inference evaluation. Bonus connection regrowth is not implemented.

## Requirements

- Python 3.10 or newer
- NumPy
- pytest (for running the tests)

The implementation does not use a deep-learning framework or an external
autodiff library.

## Install

From the repository root, create and activate a virtual environment if desired,
then install the runtime and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pytest
```

If PowerShell blocks activation, run the commands with the virtual environment's
interpreter directly, for example `.venv\Scripts\python.exe -m pytest -q
tests\test_autodiff.py`.

## Run the tests

From the repository root, run Task 1 alone, Task 4 alone, or the complete
regression suite:

```powershell
python -m pytest -q tests\test_autodiff.py
python -m pytest -q tests\test_sweep.py
python -m pytest -q tests
```

Task 1 tests check autodiff operations and gradients. Task 2 tests check
initialization, layers, AdamW updates, gradient clipping, and mini-batch
training. Task 3 tests check Taylor saliency, the cubic schedule, exact global
sparsity, binary masks, and optimizer-state protection. Task 4 tests check
pruning comparisons, CSR conversion and inference, FLOP accounting, sweep
reporting, and output files.

## Minimal usage

```python
import numpy as np

from autodiff import Tensor, mean, relu, softmax_cross_entropy

inputs = Tensor([[1.0, -2.0], [0.5, 1.5]], requires_grad=True)
weights = Tensor([[0.2, -0.3, 0.1], [0.4, 0.2, -0.5]], requires_grad=True)
targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

logits = relu(inputs @ weights)
loss = softmax_cross_entropy(logits, targets)
loss.backward()

print("loss:", loss.data.item())
print("weight gradients:", weights.grad)
```

`Tensor` values record their parent operations. Calling `backward()` on a scalar
loss visits those operations in reverse topological order and accumulates
gradients on tensors that requested them.

## Task 2 MLP training demo

Run the deterministic three-class synthetic classification example:

```powershell
python -m train.trainer
```

Each printed line is one epoch's mean cross-entropy loss and training accuracy.
The demo uses He initialization for ReLU layers, Xavier initialization for the
linear output layer, shuffled mini-batches, global gradient clipping, and Adam
with decoupled weight decay.

The core APIs are also available directly:

```python
import numpy as np

from nn import Adam, Linear, ReLU, Sequential
from train.trainer import train_classifier

rng = np.random.default_rng(7)
model = Sequential(
    Linear(2, 16, activation="relu", rng=rng),
    ReLU(),
    Linear(16, 3, activation="linear", rng=rng),
)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
history = train_classifier(model, optimizer, features, class_ids)
print(history[-1])
```

`Linear` uses He normal initialization for ReLU and Xavier uniform
initialization for tanh, sigmoid, or linear activations. Its affine computation
uses the existing `Tensor` matrix multiplication and broadcasted addition, so
the Task 1 backward engine computes weight and bias gradients. `Sequential`
composes layers and collects their trainable parameters.

`Adam` updates first and second gradient moments, applies per-entry bias
correction, and uses decoupled AdamW weight decay. Optional binary masks can be
passed to `step(masks={parameter_index: mask})`; inactive values, moments, and
bias-correction ages are cleared so a later re-enabled connection starts with
fresh optimizer state. See `task2_correctness_proof.ipynb` for an executable
masked-gradient, prune/revive, and training demonstration. This is mask
correctness support, not a pruning scheduler.

`train_classifier` shuffles data each epoch, includes the final partial batch,
computes stable cross-entropy through Task 1, clips the global gradient norm,
and returns one loss/accuracy record per epoch. Provide the same model
initialization seed and training shuffle seed to reproduce a run.

## Task 3 gradual pruning

Prune with the first-order Taylor score
`abs(weight * gradient)`. If removing a connection changes its weight by
`-weight`, the first-order loss change is approximately
`-weight * gradient`; its absolute value estimates the impact of removing that
connection.

`TaylorPruner` ranks active weights across the model's `Linear` layers and
prunes the lowest scores until it reaches the requested global sparsity.
Connections are masked in the forward pass by the existing Task 1
`apply_mask`, and pruned weights are set to exact zero. The pruner also asks
Adam to clear the corresponding moment and per-weight bias-correction state.
Pruning is monotonic: regrowth is not implemented.

`CubicPruningSchedule` keeps the initial sparsity through the warm-up, then
ramps towards the final target using
`final + (initial - final) * (1 - progress) ** 3`. Once the end step is
reached, the final target remains fixed. For example:

```python
from nn import Adam, Linear, ReLU, Sequential
from prune import CubicPruningSchedule, TaylorPruner
from train.trainer import train_classifier

rng = np.random.default_rng(7)
model = Sequential(
    Linear(2, 16, activation="relu", rng=rng),
    ReLU(),
    Linear(16, 3, activation="linear", rng=rng),
)
optimizer = Adam(model.parameters(), lr=0.01)
history = train_classifier(
    model,
    optimizer,
    features,
    class_ids,
    epochs=30,
    pruning_schedule=CubicPruningSchedule(
        final_sparsity=0.9,
        begin_step=5,
        end_step=20,
        update_frequency=1,
    ),
    pruner=TaylorPruner(model),
)
print(history[-1].sparsity)
```

The loop uses the gradients from that epoch's last mini-batch to choose
connections, then immediately applies the new masks and clears optimizer state.
`EpochMetrics.sparsity` reports the global masked fraction of linear weights.
This dense NumPy implementation demonstrates masking correctness; it does not
claim sparse-compute speedups.

## Main API

### Task 1 autodiff

- `Tensor(data, requires_grad=False)`: wraps a NumPy-compatible value.
- Arithmetic operators and `matmul`: differentiable addition, subtraction,
  multiplication, division, negation, scalar power, and matrix multiplication.
- `sum` and `mean`: differentiable reductions with optional axes and
  `keepdims`.
- `relu`, `tanh`, `sigmoid`, `gelu`, and `softmax`: differentiable activations.
- `softmax_cross_entropy(logits, targets)`: stable mean cross-entropy for
  one-hot or probability-distribution targets.
- `apply_mask(weights, mask)`: multiplies effective weights by a binary mask
  and gives masked entries zero gradient through that operation.
- `check_gradient(fn, values)`: compares a scalar function's analytical
  gradient with central finite differences and raises `AssertionError` if its
  relative error reaches the requested tolerance.

### Task 2 neural network and training

- `Parameter(data)`: a trainable leaf `Tensor`.
- `Module.parameters()` and `Module.zero_grad()`: collect parameters from
  nested layers and clear their accumulated gradients.
- `Linear(in_features, out_features, activation, rng)`: affine layer with
  activation-aware weight initialization and binary-mask support.
- `Sequential(*modules)`: compose layers and activation modules.
- `Adam(parameters, ...)`: pure-NumPy Adam with bias correction and optional
  decoupled weight decay.
- `train_classifier(...)`: seeded, shuffled mini-batch classifier training;
  returns epoch loss and accuracy.
- `clip_grad_norm(parameters, max_norm)`: clips the combined gradient norm
  before optimizer updates.

`apply_mask` remains an autodiff primitive. The Task 2 `Linear` layer reuses
it to keep masked weights out of the forward computation and give them zero
gradients.

### Task 3 pruning

- `taylor_saliency(weights, gradients)`: compute the first-order
  loss-sensitivity score for each weight.
- `CubicPruningSchedule(...)`: produce warm-up, ramp, and final sparsity
  targets and indicate when to update a mask.
- `TaylorPruner(model).step(target_sparsity, optimizer)`: enforce an exact
  global mask budget across `Linear` layers and clear Adam state for pruned
  entries.
- `train_classifier(..., pruning_schedule=..., pruner=...)`: optionally apply
  scheduled pruning after each selected epoch; omit both arguments for
  ordinary Task 2 training.

### Task 4 sparsity and inference evaluation

Run the default reproducible sweep from the repository root:

```powershell
python -m train.sweep
```

By default, this compares Taylor and magnitude pruning at the blueprint's
`0%, 50%, 75%, 85%, 90%, 95%, and 98%` sparsity targets across five paired
seeds. It writes raw trials, grouped mean/standard-deviation results, measured
CSR benchmarks, and a Pareto chart to `results\pareto_metrics.json` and
`results\pareto_curve.svg`. Options such as `--seeds`, `--targets`,
`--pruning-epochs`, `--cooldown-epochs`, and `--benchmark-repeats` support
smaller reproducible runs; use `python -m train.sweep --help` to see them.

The environment does not include an MNIST dataset or scikit-learn, so the
default sweep uses a deterministic three-class Gaussian synthetic dataset.
The results and generated claim explicitly describe that dataset and must not
be interpreted as MNIST or production accuracy.

The report includes dense FLOPs (`2 * batch * all weights`) and sparse FLOPs
(`2 * batch * active nonzero weights`), excluding biases and activations.
Its CSR path compiles each layer's stored weights once, checks output
equivalence against dense NumPy inference, and times the forward kernel
separately from CSR construction. The custom Python/NumPy CSR kernel is an
educational sparse-execution comparison, not a promise of runtime speedup;
consult the measured ratio and CSR build cost in the JSON output.

`run_pareto_sweep(SweepConfig(...))` returns the report in memory.
`write_sweep_results(report, output_dir)` writes the JSON and SVG files.
The sweep uses the existing `train_classifier`, `CubicPruningSchedule`, and
`TaylorPruner`; magnitude pruning is the comparison baseline, while Taylor
remains the default criterion for existing Task 3 calls.

## Project layout

```text
autodiff/
    __init__.py   Public Task 1 API
    tensor.py     Tensor and reverse-mode graph traversal
    ops.py        Differentiable operations, activations, loss, and masking
    checks.py     Finite-difference gradient checker
tests/
    test_autodiff.py
    test_nn.py
    test_training.py
    test_pruning.py
nn/
    module.py       Module and Parameter base classes
    layers.py       Linear, Sequential, and activation modules
    init.py         He normal and Xavier uniform initializers
    optim.py        Adam with bias correction and AdamW decay
train/
    trainer.py      Mini-batch loop, metrics, clipping, and runnable demo
    sweep.py        Task 4 multi-seed comparison, CSR inference, and reports
prune/
    criteria.py     Taylor saliency
    scheduler.py    Cubic sparsity schedule
    pruner.py       Global Taylor pruning and result metrics
tests/
    test_sweep.py   Task 4 sweeps, CSR, FLOPs, validation, and output tests
```
