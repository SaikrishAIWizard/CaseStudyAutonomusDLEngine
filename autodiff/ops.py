from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .tensor import FloatArray, Tensor


def _as_tensor(value: ArrayLike | Tensor) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def _unbroadcast(grad: FloatArray, target_shape: tuple[int, ...]) -> FloatArray:
    """Sum broadcast dimensions so a gradient matches an operand's shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    if grad.shape != target_shape:
        raise ValueError(
            f"Cannot reduce gradient shape {grad.shape} to operand shape "
            f"{target_shape}"
        )
    return grad


def _binary(
    left: ArrayLike | Tensor,
    right: ArrayLike | Tensor,
    operation: str,
) -> Tensor:
    a, b = _as_tensor(left), _as_tensor(right)
    if operation == "add":
        data = a.data + b.data
        backward = lambda grad: (
            _unbroadcast(grad, a.shape) if a.requires_grad else None,
            _unbroadcast(grad, b.shape) if b.requires_grad else None,
        )
    elif operation == "sub":
        data = a.data - b.data
        backward = lambda grad: (
            _unbroadcast(grad, a.shape) if a.requires_grad else None,
            _unbroadcast(-grad, b.shape) if b.requires_grad else None,
        )
    elif operation == "mul":
        a_data, b_data = a.data.copy(), b.data.copy()
        data = a_data * b_data
        backward = lambda grad: (
            _unbroadcast(grad * b_data, a.shape) if a.requires_grad else None,
            _unbroadcast(grad * a_data, b.shape) if b.requires_grad else None,
        )
    elif operation == "div":
        a_data, b_data = a.data.copy(), b.data.copy()
        data = a_data / b_data
        backward = lambda grad: (
            _unbroadcast(grad / b_data, a.shape) if a.requires_grad else None,
            (
                _unbroadcast(-grad * a_data / np.square(b_data), b.shape)
                if b.requires_grad
                else None
            ),
        )
    else:
        raise ValueError(f"Unsupported binary operation: {operation}")

    return Tensor(data, a.requires_grad or b.requires_grad, (a, b), operation, backward)


def add(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor:
    return _binary(a, b, "add")


def sub(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor:
    return _binary(a, b, "sub")


def mul(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor:
    return _binary(a, b, "mul")


def div(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor:
    return _binary(a, b, "div")


def neg(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    return Tensor(
        -x.data,
        x.requires_grad,
        (x,),
        "neg",
        lambda grad: (-grad if x.requires_grad else None,),
    )


def power(value: ArrayLike | Tensor, exponent: float) -> Tensor:
    if not np.isscalar(exponent):
        raise TypeError("The exponent must be a scalar")
    x = _as_tensor(value)
    x_data = x.data.copy()
    data = np.power(x_data, exponent)
    return Tensor(
        data,
        x.requires_grad,
        (x,),
        "power",
        lambda grad: (
            grad * exponent * np.power(x_data, exponent - 1)
            if x.requires_grad
            else None,
        ),
    )


def matmul(a: ArrayLike | Tensor, b: ArrayLike | Tensor) -> Tensor:
    left, right = _as_tensor(a), _as_tensor(b)
    if left.ndim == 0 or right.ndim == 0:
        raise ValueError("Matrix multiplication requires non-scalar operands")

    left_data, right_data = left.data.copy(), right.data.copy()
    data = np.matmul(left_data, right_data)
    left_vector, right_vector = left.ndim == 1, right.ndim == 1
    left_matrix = np.expand_dims(left_data, axis=-2) if left_vector else left_data
    right_matrix = np.expand_dims(right_data, axis=-1) if right_vector else right_data

    def backward(grad: FloatArray) -> tuple[FloatArray | None, ...]:
        grad_matrix = grad
        if left_vector and right_vector:
            grad_matrix = np.expand_dims(np.expand_dims(grad_matrix, -1), -1)
        elif left_vector:
            grad_matrix = np.expand_dims(grad_matrix, -2)
        elif right_vector:
            grad_matrix = np.expand_dims(grad_matrix, -1)

        grad_left = None
        if left.requires_grad:
            grad_left_matrix = np.matmul(
                grad_matrix, np.swapaxes(right_matrix, -1, -2)
            )
            if left_vector:
                grad_left_matrix = np.squeeze(grad_left_matrix, axis=-2)
            grad_left = _unbroadcast(grad_left_matrix, left.shape)

        grad_right = None
        if right.requires_grad:
            grad_right_matrix = np.matmul(
                np.swapaxes(left_matrix, -1, -2), grad_matrix
            )
            if right_vector:
                grad_right_matrix = np.squeeze(grad_right_matrix, axis=-1)
            grad_right = _unbroadcast(grad_right_matrix, right.shape)

        return grad_left, grad_right

    return Tensor(
        data,
        left.requires_grad or right.requires_grad,
        (left, right),
        "matmul",
        backward,
    )


def _normalize_axes(
    axis: int | Sequence[int] | None, ndim: int
) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = tuple(a + ndim if a < 0 else a for a in axes)
    if any(a < 0 or a >= ndim for a in normalized):
        raise ValueError(f"Reduction axis {axis} is out of bounds for ndim={ndim}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Reduction axes must be unique")
    return normalized


def _reduce(
    value: ArrayLike | Tensor,
    axis: int | Sequence[int] | None,
    keepdims: bool,
    average: bool,
) -> Tensor:
    x = _as_tensor(value)
    axes = _normalize_axes(axis, x.ndim)
    if average:
        data = np.mean(x.data, axis=axis, keepdims=keepdims)
        count = int(np.prod([x.shape[a] for a in axes], dtype=np.int64))
    else:
        data = np.sum(x.data, axis=axis, keepdims=keepdims)
        count = 1

    def backward(grad: FloatArray) -> tuple[FloatArray | None, ...]:
        if not x.requires_grad:
            return (None,)
        expanded = grad
        if not keepdims and axes:
            expanded = np.expand_dims(expanded, axis=tuple(sorted(axes)))
        contribution = np.broadcast_to(expanded, x.shape)
        if average:
            contribution = contribution / count
        return (np.array(contribution, copy=True),)

    return Tensor(
        data,
        x.requires_grad,
        (x,),
        "mean" if average else "sum",
        backward,
    )


def sum(
    value: ArrayLike | Tensor,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _reduce(value, axis, keepdims, average=False)


def mean(
    value: ArrayLike | Tensor,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _reduce(value, axis, keepdims, average=True)


def relu(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    positive = x.data > 0
    return Tensor(
        np.maximum(x.data, 0),
        x.requires_grad,
        (x,),
        "relu",
        lambda grad: (grad * positive if x.requires_grad else None,),
    )


def tanh(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    output = np.tanh(x.data)
    return Tensor(
        output,
        x.requires_grad,
        (x,),
        "tanh",
        lambda grad: (grad * (1 - np.square(output)) if x.requires_grad else None,),
    )


def sigmoid(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    output = np.empty_like(x.data)
    positive = x.data >= 0
    output[positive] = 1 / (1 + np.exp(-x.data[positive]))
    exp_x = np.exp(x.data[~positive])
    output[~positive] = exp_x / (1 + exp_x)
    return Tensor(
        output,
        x.requires_grad,
        (x,),
        "sigmoid",
        lambda grad: (
            grad * output * (1 - output) if x.requires_grad else None,
        ),
    )


def gelu(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    x_data = x.data.copy()
    coefficient = np.sqrt(2 / np.pi)
    inner = coefficient * (x_data + 0.044715 * np.power(x_data, 3))
    tanh_inner = np.tanh(inner)
    output = 0.5 * x_data * (1 + tanh_inner)
    derivative = (
        0.5 * (1 + tanh_inner)
        + 0.5
        * x_data
        * (1 - np.square(tanh_inner))
        * coefficient
        * (1 + 3 * 0.044715 * np.square(x_data))
    )
    return Tensor(
        output,
        x.requires_grad,
        (x,),
        "gelu",
        lambda grad: (grad * derivative if x.requires_grad else None,),
    )


def softmax(value: ArrayLike | Tensor) -> Tensor:
    x = _as_tensor(value)
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("Softmax requires a non-empty class dimension")
    shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def backward(grad: FloatArray) -> tuple[FloatArray | None, ...]:
        if not x.requires_grad:
            return (None,)
        projection = np.sum(grad * probabilities, axis=-1, keepdims=True)
        return (probabilities * (grad - projection),)

    return Tensor(probabilities, x.requires_grad, (x,), "softmax", backward)


def softmax_cross_entropy(
    logits: ArrayLike | Tensor, targets_one_hot: ArrayLike
) -> Tensor:
    """Return mean cross-entropy using a stable fused softmax/log-sum-exp."""
    x = _as_tensor(logits)
    targets = np.asarray(targets_one_hot, dtype=np.float64)
    if x.ndim < 2 or x.shape[-1] == 0:
        raise ValueError("Logits must have a batch dimension and non-empty classes")
    if targets.shape != x.shape:
        raise ValueError(
            f"Target shape {targets.shape} must match logits shape {x.shape}"
        )
    if not np.all(np.isfinite(x.data)) or not np.all(np.isfinite(targets)):
        raise ValueError("Logits and targets must contain only finite values")
    if np.any(targets < 0) or not np.allclose(
        np.sum(targets, axis=-1), 1.0, rtol=0.0, atol=1e-7
    ):
        raise ValueError("Each target row must be a probability distribution")

    shifted = x.data - np.max(x.data, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    log_normalizer = np.log(np.sum(exp_values, axis=-1)) + np.max(
        x.data, axis=-1
    )
    per_row_loss = log_normalizer - np.sum(targets * x.data, axis=-1)
    row_count = int(np.prod(x.shape[:-1], dtype=np.int64))
    loss = np.sum(per_row_loss) / row_count

    def backward(grad: FloatArray) -> tuple[FloatArray | None, ...]:
        if not x.requires_grad:
            return (None,)
        return (((probabilities - targets) / row_count) * grad,)

    return Tensor(loss, x.requires_grad, (x,), "softmax_cross_entropy", backward)


def apply_mask(value: ArrayLike | Tensor, mask: ArrayLike) -> Tensor:
    """Apply a fixed binary mask and block gradients at masked positions."""
    x = _as_tensor(value)
    mask_data = np.asarray(mask)
    if not np.all(np.isin(mask_data, (0, 1))):
        raise ValueError("A weight mask must contain only zeros and ones")
    try:
        expanded_mask = np.broadcast_to(mask_data, x.shape).astype(np.float64)
    except ValueError as exc:
        raise ValueError(
            f"Mask shape {mask_data.shape} cannot be broadcast to tensor shape "
            f"{x.shape}"
        ) from exc
    saved_mask = expanded_mask.copy()
    return Tensor(
        x.data * saved_mask,
        x.requires_grad,
        (x,),
        "mask",
        lambda grad: (grad * saved_mask if x.requires_grad else None,),
    )
