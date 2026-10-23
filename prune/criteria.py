from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def taylor_saliency(
    weights: ArrayLike, gradients: ArrayLike
) -> NDArray[np.float64]:
    """Estimate each connection's loss impact with |weight * gradient|.

    Removing a weight changes it by -weight. The first-order Taylor loss
    change is therefore approximately -weight * gradient; its absolute value
    gives the saliency used to rank connections for pruning.
    """
    weight_values = np.asarray(weights, dtype=np.float64)
    gradient_values = np.asarray(gradients, dtype=np.float64)
    if weight_values.shape != gradient_values.shape:
        raise ValueError(
            f"Weight shape {weight_values.shape} does not match gradient shape "
            f"{gradient_values.shape}"
        )
    if not np.all(np.isfinite(weight_values)):
        raise ValueError("Weights must contain only finite values")
    if not np.all(np.isfinite(gradient_values)):
        raise ValueError("Gradients must contain only finite values")
    return np.abs(weight_values * gradient_values)
