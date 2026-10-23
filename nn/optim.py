from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np
from numpy.typing import ArrayLike

from .module import Parameter


def _finite_number(name: str, value: float, *, minimum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


class Adam:
    """Adam optimizer with bias correction and decoupled AdamW weight decay."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = tuple(params)
        if not self.params:
            raise ValueError("Adam requires at least one parameter")
        if not all(isinstance(parameter, Parameter) for parameter in self.params):
            raise TypeError("Adam parameters must be Parameter instances")
        if len({id(parameter) for parameter in self.params}) != len(self.params):
            raise ValueError("Adam parameters must not contain duplicates")

        self.lr = _finite_number("lr", lr, minimum=0.0)
        if self.lr == 0:
            raise ValueError("lr must be greater than zero")
        self.beta1 = _finite_number("beta1", beta1, minimum=0.0)
        self.beta2 = _finite_number("beta2", beta2, minimum=0.0)
        if self.beta1 >= 1 or self.beta2 >= 1:
            raise ValueError("beta1 and beta2 must be less than one")
        self.eps = _finite_number("eps", eps, minimum=0.0)
        if self.eps == 0:
            raise ValueError("eps must be greater than zero")
        self.weight_decay = _finite_number(
            "weight_decay", weight_decay, minimum=0.0
        )

        self.t = 0
        self.m = [np.zeros_like(parameter.data) for parameter in self.params]
        self.v = [np.zeros_like(parameter.data) for parameter in self.params]
        self.step_counts = [
            np.zeros_like(parameter.data, dtype=np.int64) for parameter in self.params
        ]

    def zero_grad(self) -> None:
        """Clear gradients before the next training batch."""
        for parameter in self.params:
            parameter.grad = None

    def apply_masks(self, masks: Mapping[int, ArrayLike]) -> None:
        """Enforce binary parameter masks and clear inactive Adam state."""
        self._apply_prepared_masks(self._prepare_masks(masks))

    def step(self, masks: Mapping[int, ArrayLike] | None = None) -> None:
        """Apply one AdamW update, optionally freezing masked parameter entries.

        Masks are keyed by parameter index and must contain only zeroes and ones.
        Inactive entries are set to zero and their moment buffers and per-entry
        bias-correction ages are cleared, so a subsequently re-enabled entry
        starts with clean optimizer state.
        """
        prepared_masks = self._prepare_masks(masks)

        for parameter in self.params:
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=np.float64)
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"Gradient shape {gradient.shape} does not match parameter "
                    f"shape {parameter.shape}"
                )
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Adam received a non-finite gradient")

        self._apply_prepared_masks(prepared_masks)

        self.t += 1

        for index, parameter in enumerate(self.params):
            if parameter.grad is None:
                continue
            gradient = np.asarray(parameter.grad, dtype=np.float64)

            mask = prepared_masks.get(index)
            if mask is not None:
                gradient = gradient * mask
                active = mask != 0
            else:
                active = np.ones(parameter.shape, dtype=bool)

            self.m[index] = (
                self.beta1 * self.m[index] + (1.0 - self.beta1) * gradient
            )
            self.v[index] = (
                self.beta2 * self.v[index]
                + (1.0 - self.beta2) * np.square(gradient)
            )
            self.step_counts[index][active] += 1

            entry_correction1 = 1.0 - np.power(
                self.beta1, self.step_counts[index]
            )
            entry_correction2 = 1.0 - np.power(
                self.beta2, self.step_counts[index]
            )
            corrected_m = np.divide(
                self.m[index],
                entry_correction1,
                out=np.zeros_like(self.m[index]),
                where=active,
            )
            corrected_v = np.divide(
                self.v[index],
                entry_correction2,
                out=np.zeros_like(self.v[index]),
                where=active,
            )
            update = corrected_m / (np.sqrt(corrected_v) + self.eps)

            if self.weight_decay:
                decay = self.weight_decay * parameter.data
                if mask is not None:
                    decay = decay * mask
                parameter.data -= self.lr * decay
            parameter.data -= self.lr * update
            if mask is not None:
                parameter.data[mask == 0] = 0.0

    def _prepare_masks(
        self, masks: Mapping[int, ArrayLike] | None
    ) -> dict[int, np.ndarray]:
        prepared: dict[int, np.ndarray] = {}
        if masks is None:
            return prepared
        if not isinstance(masks, Mapping):
            raise TypeError("masks must map parameter indices to binary masks")
        for index, mask in masks.items():
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(self.params)
            ):
                raise ValueError(f"Invalid parameter index in masks: {index}")
            mask_array = np.asarray(mask)
            if not np.all(np.isin(mask_array, (0, 1))):
                raise ValueError("Optimizer masks must contain only zeros and ones")
            try:
                prepared[index] = np.broadcast_to(
                    mask_array, self.params[index].shape
                ).astype(np.float64, copy=True)
            except ValueError as exc:
                raise ValueError(
                    f"Mask shape {mask_array.shape} cannot be broadcast to "
                    f"parameter shape {self.params[index].shape}"
                ) from exc
        return prepared

    def _apply_prepared_masks(self, masks: Mapping[int, np.ndarray]) -> None:
        for index, mask in masks.items():
            inactive = mask == 0
            self.params[index].data[inactive] = 0.0
            self.m[index][inactive] = 0.0
            self.v[index][inactive] = 0.0
            self.step_counts[index][inactive] = 0
