__all__ = [
    "trunc_powerlaw_weights",
]

from typing import Optional

import numpy as np


def trunc_powerlaw_weights(alpha: float, v_min: int, v_max: int) -> np.ndarray[float]:
    if alpha < 1:
        raise ValueError(f"Parameter 'alpha' should be greater than or equal to 1. Got: {alpha}.")

    if v_min < 1 or v_min > v_max:
        raise ValueError(f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}.")

    return np.array([1 / i**alpha for i in range(v_min, v_max)])


def sample_trunc_powerlaw(
    v_min: int, v_max: int, n: int, *, w: Optional[np.ndarray] = None, alpha: Optional[float] = None
) -> np.ndarray:

    if sum(i is not None for i in [alpha, w]) != 1:
        raise ValueError("Only one of the params 'alpha' and 'w' can be specified")

    if alpha and alpha < 1:
        raise ValueError(f"Parameter 'alpha' should be greater than or equal to 1. Got: {alpha}.")

    if v_min < 1 or v_min > v_max:
        raise ValueError(f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}.")

    if n < 0:
        raise ValueError("Parameter 'n' must be a positive integer")

    w = w or np.array([1 / i**alpha for i in range(v_min, v_max + 1)])
    return np.random.choice(range(v_min, v_max + 1), size=n, p=w)
