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

import numpy as np

from abcd_graph_generator import ABCDParams


class GraphGenModel(ABC):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        self.params = params
        self.clusters = clusters

        self.cluster_weight = np.zeros(len(params.s))
        for i, weight in enumerate(params.w):
            self.cluster_weight[clusters[i]] += weight

        self.total_weight = self.cluster_weight.sum()

    @abstractmethod
    def get_edges(self) -> Set[Tuple[int, int]]:
        ...
