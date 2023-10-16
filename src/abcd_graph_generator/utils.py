__all__ = [
    "get_mul",
    "get_tabu",
    "randround",
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
        phi = 1 - sum((sl / (n - s0)) ** 2 for sl in params.s[2:]) * (n - s0) * params.xi / ((n - s0) * params.xi + s0)
    else:
        phi = 1 - sum((sl / n) ^ 2 for sl in params.s)

    return 1.0 - params.xi * phi


def get_tabu(params: ABCDParams) -> np.ndarray:
    nout = params.s[0]
    n = len(params.w)
    L = sum(map(lambda d: min(1.0, params.xi * d), params.w))
    threshold = L + nout - L * nout / n - 1.0
    idx = np.argmax(params.w <= threshold)  # index of the first elem of w that is less or equal to the threshold

    if len(params.w[idx:]) < nout:
        raise ValueError("Not enough nodes feasible for classification as outliers")

    return np.random.choice(range(idx, n + 1), nout, replace=False)


def randround(x: Union[float, int]) -> int:
    d = np.floor(x)
    return d + int(np.random.random() < x - d)
