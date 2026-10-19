from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Backward = Callable[[FloatArray], tuple[FloatArray | None, ...]]


class Tensor:
    """A NumPy value and its reverse-mode automatic differentiation history."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        parents: Iterable[Tensor] = (),
        op: str = "",
        backward: Backward | None = None,
    ) -> None:
        self.data: FloatArray = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = requires_grad
        self.grad: FloatArray | None = None
        self.parents = tuple(parents)
        self.op = op
        self._backward: Backward = backward or (lambda grad: ())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Accumulate gradients from this tensor through its computation graph."""
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "An implicit gradient can only be created for a scalar output"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=np.float64)
            if seed.shape != self.shape:
                raise ValueError(
                    f"Gradient shape {seed.shape} does not match tensor shape "
                    f"{self.shape}"
                )

        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))

        local_grads: dict[Tensor, FloatArray] = {self: seed.copy()}
        for node in reversed(topo):
            upstream = local_grads.get(node)
            if upstream is None:
                continue

            parent_grads = node._backward(upstream)
            if len(parent_grads) != len(node.parents):
                raise RuntimeError(
                    f"Backward function for {node.op or 'tensor'} returned "
                    f"{len(parent_grads)} gradients for {len(node.parents)} parents"
                )

            for parent, parent_grad in zip(node.parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                contribution = np.asarray(parent_grad, dtype=np.float64)
                if contribution.shape != parent.shape:
                    raise RuntimeError(
                        f"Backward function for {node.op or 'tensor'} returned "
                        f"gradient shape {contribution.shape}; expected "
                        f"{parent.shape}"
                    )
                if parent in local_grads:
                    local_grads[parent] += contribution
                else:
                    local_grads[parent] = contribution.copy()

        for node, contribution in local_grads.items():
            if node.grad is None:
                node.grad = contribution.copy()
            else:
                node.grad += contribution

    def zero_grad(self) -> None:
        """Clear this tensor and all of its ancestors' accumulated gradients."""
        stack = [self]
        visited: set[Tensor] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            node.grad = None
            stack.extend(node.parents)

    def __add__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import add

        return add(self, other)

    def __radd__(self, other: ArrayLike | Tensor) -> Tensor:
        return self + other

    def __sub__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import sub

        return sub(self, other)

    def __rsub__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import sub

        return sub(other, self)

    def __mul__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import mul

        return mul(self, other)

    def __rmul__(self, other: ArrayLike | Tensor) -> Tensor:
        return self * other

    def __truediv__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import div

        return div(self, other)

    def __rtruediv__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import div

        return div(other, self)

    def __neg__(self) -> Tensor:
        from .ops import neg

        return neg(self)

    def __pow__(self, exponent: float) -> Tensor:
        from .ops import power

        return power(self, exponent)

    def __matmul__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: ArrayLike | Tensor) -> Tensor:
        from .ops import matmul

        return matmul(other, self)

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, requires_grad={self.requires_grad}, "
            f"grad={self.grad!r})"
        )
