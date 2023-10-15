__all__ = [
    "trunc_powerlaw_weights",
]

import numpy as np


def trunc_powerlaw_weights(alpha: float, v_min: int, v_max: int) -> np.ndarray[float]:
    if alpha < 1:
        raise ValueError(f"Parameter 'alpha' should be greater or equal to 1. Got: {alpha}.")

    if v_min < 1 or v_min > v_max:
        raise ValueError(f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}.")

    return np.array([1 / i**alpha for i in range(v_min, v_max)])
