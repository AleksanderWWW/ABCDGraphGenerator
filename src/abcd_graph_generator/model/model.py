__all__ = [
    "GraphGenModel",
]

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Set,
    Tuple,
)


class GraphGenModel(ABC):
    @abstractmethod
    def get_edges(self) -> Set[Tuple[int, int]]:
        ...
