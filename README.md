# Aqua Case Study - Tasks 1 and 2

This repository implements the first two parts of the Aqua Platform challenge:
a reverse-mode automatic differentiation engine and a small NumPy-only MLP
with AdamW training. Self-pruning schedules, connection regrowth, and Pareto
sweeps from Parts 3-4 are not implemented.

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
run task1 only
python -m pytest -q tests\test_autodiff.py
Run both the Task 1 and Task 2 tests from the repository root:

```powershell
python -m pytest -q tests
```

Task 1 tests check the autodiff operations and gradients. Task 2 tests check
initialization, layers, AdamW updates, masking state, gradient clipping, and
mini-batch training.

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
gradients. The optimizer also clears moments for inactive entries; no pruning
schedule or regrowth logic is included.

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
nn/
    module.py       Module and Parameter base classes
    layers.py       Linear, Sequential, and activation modules
    init.py         He normal and Xavier uniform initializers
    optim.py        Adam with bias correction and AdamW decay
train/
    trainer.py      Mini-batch loop, metrics, clipping, and runnable demo
```
