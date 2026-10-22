from __future__ import annotations

from collections.abc import Iterable

from numpy.typing import ArrayLike, NDArray

from autodiff import Tensor


class Parameter(Tensor):
    """A leaf tensor that a neural-network optimizer can update."""

    def __init__(self, data: ArrayLike) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for callable neural-network layers and containers."""

    def __call__(self, inputs: ArrayLike | Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: ArrayLike | Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.forward() must be implemented"
        )

    def parameters(self) -> list[Parameter]:
        """Return this module's parameters, including nested child modules."""
        found: list[Parameter] = []
        seen_parameters: set[int] = set()
        seen_modules: set[int] = set()

        def visit(value: object) -> None:
            if isinstance(value, Parameter):
                if id(value) not in seen_parameters:
                    seen_parameters.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                if id(value) in seen_modules:
                    return
                seen_modules.add(id(value))
                for child in vars(value).values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(self)
        return found

    def zero_grad(self) -> None:
        """Clear accumulated gradients on all parameters in this module."""
        for parameter in self.parameters():
            parameter.grad = None

    def parameter_masks(self) -> dict[Parameter, NDArray[np.float64]]:
        """Return optional parameter masks; ordinary modules have none."""
        return {}


def validate_modules(modules: Iterable[Module]) -> tuple[Module, ...]:
    """Materialize and validate child modules for a Sequential container."""
    children = tuple(modules)
    if not all(isinstance(child, Module) for child in children):
        raise TypeError("Sequential accepts only Module instances")
    return children
