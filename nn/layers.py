from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from autodiff import Tensor, apply_mask, relu, sigmoid, tanh

from .init import he_normal, xavier_uniform
from .module import Module, Parameter, validate_modules


class Linear(Module):
    """An affine layer; `activation` selects its weight initializer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if activation not in {"relu", "tanh", "sigmoid", "linear"}:
            raise ValueError(
                "activation must be 'relu', 'tanh', 'sigmoid', or 'linear'"
            )

        shape = (in_features, out_features)
        weight_data = (
            he_normal(shape, rng)
            if activation == "relu"
            else xavier_uniform(shape, rng)
        )
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.W = Parameter(weight_data)
        self.b = Parameter([0.0] * out_features)
        self.mask = np.ones(shape, dtype=np.float64)

    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        values = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        if values.ndim == 0 or values.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input's last dimension to be {self.in_features}; "
                f"received shape {values.shape}"
            )
        return values @ apply_mask(self.W, self.mask) + self.b

    def parameter_masks(self) -> dict[Parameter, np.ndarray]:
        return {self.W: self.mask}


class Sequential(Module):
    """Apply child modules from first to last."""

    def __init__(self, *modules: Module) -> None:
        self.modules: tuple[Module, ...] = validate_modules(modules)

    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        output = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        for module in self.modules:
            output = module(output)
        return output

    def parameter_masks(self) -> dict[Parameter, np.ndarray]:
        masks: dict[Parameter, np.ndarray] = {}
        for module in self.modules:
            masks.update(module.parameter_masks())
        return masks


class ReLU(Module):
    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        return relu(inputs)


class Tanh(Module):
    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        return tanh(inputs)


class Sigmoid(Module):
    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        return sigmoid(inputs)
