from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_step(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_sparsity(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


@dataclass(frozen=True)
class CubicPruningSchedule:
    """Ramp sparsity from an initial to final target with a cubic schedule."""

    final_sparsity: float = 0.9
    begin_step: int = 0
    end_step: int = 100
    initial_sparsity: float = 0.0
    update_frequency: int = 1

    def __post_init__(self) -> None:
        initial = _validate_sparsity("initial_sparsity", self.initial_sparsity)
        final = _validate_sparsity("final_sparsity", self.final_sparsity)
        begin = _validate_step("begin_step", self.begin_step)
        end = _validate_step("end_step", self.end_step)
        frequency = _validate_step("update_frequency", self.update_frequency)

        if final < initial:
            raise ValueError(
                "final_sparsity must be greater than or equal to initial_sparsity"
            )
        if end <= begin:
            raise ValueError("end_step must be greater than begin_step")
        if frequency == 0:
            raise ValueError("update_frequency must be a positive integer")

    def sparsity_at(self, step: int) -> float:
        """Return the cubic target sparsity at a training step."""
        current_step = _validate_step("step", step)
        if current_step <= self.begin_step:
            return float(self.initial_sparsity)
        if current_step >= self.end_step:
            return float(self.final_sparsity)

        progress = (current_step - self.begin_step) / (
            self.end_step - self.begin_step
        )
        remaining = 1.0 - progress
        return float(
            self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity) * remaining**3
        )

    def should_update(self, step: int) -> bool:
        """Whether the pruning target should be applied at this step."""
        current_step = _validate_step("step", step)
        if not self.begin_step <= current_step <= self.end_step:
            return False
        return (
            (current_step - self.begin_step) % self.update_frequency == 0
            or current_step == self.end_step
        )
