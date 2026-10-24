from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .tensor import Tensor


def check_gradient(
    fn: Callable[[Tensor], Tensor],
    values: ArrayLike,
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> float:
    """Check a scalar function's derivative with central finite differences.

    Returns the relative error and raises AssertionError when it reaches `tol`.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Finite-difference epsilon must be positive")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("Gradient-check tolerance must be positive")

    values_array = np.array(values, dtype=np.float64, copy=True)
    variable = Tensor(values_array, requires_grad=True)
    loss = fn(variable)
    if not isinstance(loss, Tensor):
        raise TypeError("The checked function must return a Tensor")
    if loss.data.size != 1:
        raise ValueError("The checked function must return a scalar Tensor")
    loss.backward()
    if variable.grad is None:
        raise RuntimeError("The checked function produced no input gradient")
    analytic = variable.grad.copy()

    numeric = np.zeros_like(values_array)
    for index in np.ndindex(values_array.shape):
        plus = values_array.copy()
        minus = values_array.copy()
        plus[index] += eps
        minus[index] -= eps
        plus_loss = fn(Tensor(plus))
        minus_loss = fn(Tensor(minus))
        if not isinstance(plus_loss, Tensor) or not isinstance(minus_loss, Tensor):
            raise TypeError("The checked function must return a Tensor")
        if plus_loss.data.size != 1 or minus_loss.data.size != 1:
            raise ValueError("The checked function must return a scalar Tensor")
        numeric[index] = (plus_loss.data.item() - minus_loss.data.item()) / (2 * eps)

    denominator = np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-8
    relative_error = float(np.linalg.norm(analytic - numeric) / denominator)
    if not np.isfinite(relative_error) or relative_error >= tol:
        raise AssertionError(
            f"Gradient check failed: relative error {relative_error:.2e} "
            f"is not below tolerance {tol:.2e}"
        )
    return relative_error
