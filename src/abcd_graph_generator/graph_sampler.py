__all__ = [
    "ABCDParams",
    "randround",
]

import math
from dataclasses import dataclass
from typing import (
    Optional,
    Union,
)

import numpy as np


@dataclass(slots=True)
class ABCDParams:
    w: np.ndarray
    s: np.ndarray
    mu: Optional[float] = None
    xi: Optional[float] = None
    is_CL: bool = False
    is_local: bool = False
    has_outliers: bool = False

    def __post_init__(self) -> None:
        self.validate_w_and_s()

        self.validate_mu()

        self.validate_xi()

        self.validate_xi_and_mu()

        self.handle_if_outliers()

    def validate_w_and_s(self):
        if len(self.w) != self.s.sum():
            raise ValueError("Inconsistent data")

    def validate_mu(self):
        if not self.mu:
            return

        if self.mu < 0 or self.mu > 1:
            raise ValueError("Inconsistent data on μ")

    def validate_xi(self):
        if not self.xi:
            return

        if self.is_local:
            raise ValueError("when ξ is provided local model is not allowed")

        if self.xi < 0 or self.xi > 1:
            raise ValueError("Inconsistent data on ξ")

    def validate_xi_and_mu(self):
        if not (self.mu or self.xi):
            raise ValueError("Inconsistent data: either μ or ξ must be provided")

        if self.mu and self.xi:
            raise ValueError("Inconsistent data: only μ or ξ may be provided")

    def handle_if_outliers(self):
        if not self.has_outliers:
            return

        # TODO sorting logic


def randround(x: Union[float, int]) -> int:
    d = math.floor(x)
    return d + int(np.random.random() < x - d)
