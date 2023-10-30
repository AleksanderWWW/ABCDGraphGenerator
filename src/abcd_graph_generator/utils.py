__all__ = [
    "randround",
    "is_odd",
    "is_even",
    "minmax",
]

from typing import (
    Iterable,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

"""Module to host helper functions for calculations."""


T = TypeVar("T")


def randround(x: Union[float, int]) -> int:
    d = np.floor(x)
    return d + int(np.random.random() < x - d)


def is_odd(x: int) -> bool:
    return x % 2 != 0


def is_even(x: int) -> bool:
    return x % 2 == 0


def minmax(__iter: Iterable[T]) -> Tuple[T, T]:
    if isinstance(__iter, np.ndarray):
        return __iter.min(), __iter.max()

    return min(__iter), max(__iter)
