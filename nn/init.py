from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if (
        len(shape) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) for size in shape)
        or any(size <= 0 for size in shape)
    ):
        raise ValueError("Weight shape must contain two positive integers")
    return shape


def he_normal(
    shape: tuple[int, int], rng: np.random.Generator | None = None
) -> FloatArray:
    """Sample ReLU weights with standard deviation sqrt(2 / fan_in)."""
    fan_in, _ = _validate_shape(shape)
    generator = rng if rng is not None else np.random.default_rng()
    return generator.normal(0.0, math.sqrt(2.0 / fan_in), size=shape)


def xavier_uniform(
    shape: tuple[int, int], rng: np.random.Generator | None = None
) -> FloatArray:
    """Sample weights uniformly within sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = _validate_shape(shape)
    generator = rng if rng is not None else np.random.default_rng()
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return generator.uniform(-bound, bound, size=shape)
