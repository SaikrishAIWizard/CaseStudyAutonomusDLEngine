from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from nn.layers import Linear
from nn.module import Module
from nn.optim import Adam

from .criteria import taylor_saliency


@dataclass(frozen=True)
class PruningResult:
    """Connection counts and sparsity after one pruning update."""

    target_sparsity: float
    achieved_sparsity: float
    pruned_connections: int
    active_connections: int
    total_connections: int


def _linear_layers(model: Module) -> tuple[Linear, ...]:
    found: list[Linear] = []
    seen_modules: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, Linear):
            if id(value) not in seen_modules:
                seen_modules.add(id(value))
                found.append(value)
            return
        if isinstance(value, Module):
            if id(value) in seen_modules:
                return
            seen_modules.add(id(value))
            for child in vars(value).values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(model)
    return tuple(found)


class TaylorPruner:
    """Globally prune Linear-layer weights using first-order Taylor saliency."""

    def __init__(self, model: Module) -> None:
        if not isinstance(model, Module):
            raise TypeError("model must be a Module")
        self.layers = _linear_layers(model)
        if not self.layers:
            raise ValueError("model must contain at least one Linear layer")
        self._validate_masks()

    @property
    def sparsity(self) -> float:
        total = sum(layer.mask.size for layer in self.layers)
        active = sum(int(np.count_nonzero(layer.mask)) for layer in self.layers)
        return 1.0 - active / total

    def step(
        self,
        target_sparsity: float,
        optimizer: Adam,
        criterion: Literal["taylor", "magnitude"] = "taylor",
    ) -> PruningResult:
        """Prune low-score active weights to a global sparsity target.

        Existing masked connections stay pruned. The Adam optimizer is told
        about the updated masks immediately, which clears masked weights and
        their moment and bias-correction state before another update can occur.
        Taylor saliency is the default; magnitude is provided as a baseline.
        """
        if not isinstance(target_sparsity, (int, float)) or isinstance(
            target_sparsity, bool
        ):
            raise TypeError("target_sparsity must be a number")
        target = float(target_sparsity)
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_sparsity must be between zero and one")
        if not isinstance(optimizer, Adam):
            raise TypeError("optimizer must be an Adam optimizer")
        if criterion not in ("taylor", "magnitude"):
            raise ValueError("criterion must be 'taylor' or 'magnitude'")

        self._validate_masks()
        optimizer_indices = {
            id(parameter): index for index, parameter in enumerate(optimizer.params)
        }
        missing = [layer for layer in self.layers if id(layer.W) not in optimizer_indices]
        if missing:
            raise ValueError("optimizer must contain every pruned layer's weights")

        scores: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for layer in self.layers:
            if criterion == "taylor" and layer.W.grad is None:
                raise RuntimeError(
                    "Taylor pruning requires a gradient for every Linear weight"
                )
            if criterion == "taylor":
                score = taylor_saliency(layer.W.data, layer.W.grad)
            else:
                score = np.abs(layer.W.data)
            scores.append(score.reshape(-1))
            masks.append(layer.mask.reshape(-1).copy())

        total_connections = sum(mask.size for mask in masks)
        current_masks = np.concatenate(masks)
        current_zeros = total_connections - int(np.count_nonzero(current_masks))
        target_zeros = int(total_connections * target)
        if target_zeros < current_zeros:
            raise ValueError(
                "target_sparsity is below current sparsity; TaylorPruner does "
                "not regrow connections"
            )

        prune_count = target_zeros - current_zeros
        all_scores = np.concatenate(scores)
        active_indices = np.flatnonzero(current_masks)
        if prune_count:
            active_order = np.argsort(
                all_scores[active_indices], kind="stable"
            )
            current_masks[active_indices[active_order[:prune_count]]] = 0.0

        offset = 0
        parameter_masks: dict[int, np.ndarray] = {}
        for layer, layer_mask in zip(self.layers, masks):
            next_offset = offset + layer_mask.size
            layer.mask[...] = current_masks[offset:next_offset].reshape(layer.mask.shape)
            layer.W.data[...] *= layer.mask
            parameter_masks[optimizer_indices[id(layer.W)]] = layer.mask
            offset = next_offset

        optimizer.apply_masks(parameter_masks)
        active_connections = int(np.count_nonzero(current_masks))
        return PruningResult(
            target_sparsity=target,
            achieved_sparsity=1.0 - active_connections / total_connections,
            pruned_connections=prune_count,
            active_connections=active_connections,
            total_connections=total_connections,
        )

    def _validate_masks(self) -> None:
        for layer in self.layers:
            if layer.mask.shape != layer.W.shape:
                raise ValueError(
                    f"Mask shape {layer.mask.shape} does not match weight shape "
                    f"{layer.W.shape}"
                )
            if not np.all(np.isin(layer.mask, (0, 1))):
                raise ValueError("Pruning masks must contain only zeros and ones")
