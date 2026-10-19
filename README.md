# Aqua Case Study - Task 1

This repository currently implements **Part 1 only** of the Aqua Platform
challenge: a small reverse-mode automatic differentiation engine built with
Python and NumPy. The MLP and optimizer, self-pruning training loop, and Pareto
sweeps from Parts 2-4 are not implemented.

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

## Run the Task 1 tests

Run the complete Task 1 verification suite from the repository root:

```powershell
python -m pytest -q tests\test_autodiff.py
```

The tests check operation gradients against central finite differences, graph
gradient accumulation, broadcasting, reductions, matrix multiplication,
activations, stable softmax cross-entropy, and masked-weight behavior.

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

## Main API

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

`apply_mask` is an autodiff primitive, not a training-time pruning manager. It
does not implement optimizer updates or reset momentum/Adam buffers when a
connection is pruned and later revived. That optimizer-state behavior must be
implemented and tested alongside the optimizer in Part 2.

## Project layout

```text
autodiff/
    __init__.py   Public Task 1 API
    tensor.py     Tensor and reverse-mode graph traversal
    ops.py        Differentiable operations, activations, loss, and masking
    checks.py     Finite-difference gradient checker
tests/
    test_autodiff.py
```
