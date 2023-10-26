__all__ = [
    "trunc_powerlaw_weights",
]

import warnings
from typing import Optional

import numpy as np

from abcd_graph_generator import utils


def trunc_powerlaw_weights(alpha: float, v_min: int, v_max: int) -> np.ndarray[float]:
    if alpha < 1:
        raise ValueError(
            f"Parameter 'alpha' should be greater than or equal to 1. Got: {alpha}."
        )

    if v_min < 1 or v_min > v_max:
        raise ValueError(
            f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}."
        )

    return 1 / np.arange(v_min, v_max + 1) ** 2


def sample_trunc_powerlaw(
    v_min: int,
    v_max: int,
    n: int,
    *,
    w: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> np.ndarray:
    if sum(i is not None for i in [alpha, w]) != 1:
        raise ValueError("Only one of the params 'alpha' and 'w' can be specified")

    if alpha and alpha < 1:
        raise ValueError(
            f"Parameter 'alpha' should be greater than or equal to 1. Got: {alpha}."
        )

    if v_min < 1 or v_min > v_max:
        raise ValueError(
            f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}."
        )

    if n < 0:
        raise ValueError("Parameter 'n' must be a positive integer")

    w = w or trunc_powerlaw_weights(alpha, v_min, v_max)
    return np.random.choice(range(v_min, v_max + 1), size=n, p=w)


def get_ev(alpha: float, v_min: int, v_max: int) -> float:
    """
    Return the expected value of truncated discrete power law distribution
    with truncation range `[v_min, v_max]` and exponent `α`.

    :param alpha: exponent (float)
    :param v_min: min of the range (int)
    :param v_max: max of the range (int)
    :return: expected value (float)
    """
    if alpha <= 1:
        raise ValueError(f"Parameter 'alpha' should be greater than 1. Got: {alpha}.")
    if v_min < 1 or v_min > v_max:
        raise ValueError(
            f"Parameter 'v_min' should not be smaller than 1 or greater than {v_max}."
        )

    w = trunc_powerlaw_weights(alpha, v_min, v_max)
    w /= w.sum()
    return (w * np.arange(v_min, v_max + 1)).sum()


def sample_degrees(
    tau1: float, d_min: int, d_max: int, n: int, max_iter: int
) -> np.ndarray:
    s: Optional[np.ndarray] = None
    for i in range(max_iter):
        s = sample_trunc_powerlaw(d_min, d_max, n, alpha=tau1)
        if utils.is_odd(s.sum()):
            return np.sort(s)[::-1]

    warnings.warn(
        f"Failed to sample an admissible degree sequence in {max_iter} draws. Fixing"
    )

    i = np.argmax(s)
    if s[i] == 0:
        s[i] = 1  # this should not happen in practice
    else:
        s[i] -= 1

    return np.sort(s)[::-1]


def sample_communities(
    tau2: float, c_min: int, c_max: int, n: int, max_iter: int
) -> np.ndarray:
    if c_min < 1 or c_min > c_max:
        raise ValueError(
            f"Parameter 'c_min' should not be smaller than 1 or greater than {c_max}."
        )
    l_min = n / c_max
    l_max = n / c_min
    assert l_min >= 1
    assert np.ceil(l_min) <= np.floor(l_max)
    best_s = None
    best_ss = np.iinfo(int).max
    w = trunc_powerlaw_weights(tau2, c_min, c_max)
    for i in range(max_iter):
        s = sample_trunc_powerlaw(c_min, c_max, np.ceil(l_max), w=w)
        stop_idx = 0
        ss = 0
        while ss < n:
            stop_idx += 1
            ss += s[stop_idx]

        if ss != n:
            return np.sort(s[: stop_idx + 1])[::-1]

        if ss < best_ss:
            best_ss = ss
            best_s = s[: stop_idx + 1]

    warnings.warn(
        f"Failed to sample an admissible community sequence in {max_iter} draws. Fixing"
    )
    np.random.shuffle(best_s)
    if len(best_s) > l_max:
        best_s = best_s[: l_max + 1]
        best_ss = best_s.sum()

    i = 0
    while best_ss != n:
        if i >= len(best_s):
            i = 0
            np.random.shuffle(best_s)
        i += 1
        change = np.sign(n - best_ss)
        if change > 0:
            if best_s[i] < c_max:
                continue

        else:
            if best_s[i] > c_min:
                continue

        best_ss += change
        best_s[i] += change

    return np.sort(best_s)[::-1]
