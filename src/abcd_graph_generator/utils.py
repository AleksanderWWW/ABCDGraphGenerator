__all__ = [
    "get_mul",
]

"""Module to host helper functions for calculations."""

from typing import Union

import numpy as np

from abcd_graph_generator.abcd_params import ABCDParams


def get_mul(params: ABCDParams) -> float:
    if not params.xi:
        return 1.0 - params.mu

    n = len(params.w)
    if params.has_outliers:
        s0 = params.s[0]
        phi = 1.0 - sum((sl / (n - s0)) ^ 2 for sl in params.s[2:]) * (n - s0) * params.xi / ((n - s0) * params.xi + s0)
    else:
        phi = 1.0 - sum((sl / n) ^ 2 for sl in params.s)

    return 1.0 - params.xi * phi


def randround(x: Union[float, int]) -> int:
    d = np.floor(x)
    return d + int(np.random.random() < x - d)
