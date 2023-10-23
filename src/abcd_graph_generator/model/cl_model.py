__all__ = [
    "cl_model",
]

import random
from typing import (
    Set,
    Tuple,
)

import numpy as np

from abcd_graph_generator import (
    ABCDParams,
    utils,
)
from abcd_graph_generator.model.model import GraphGenModel
from abcd_graph_generator.utils import minmax


def cl_model(clusters: np.ndarray, params: ABCDParams) -> Set[Tuple[int, int]]:
    model = (
        LocalCLModel(params, clusters)
        if params.is_local
        else GlobalCLModel(params, clusters)
    )

    return model.get_edges()


class CLModel(GraphGenModel):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        if params.has_outliers or not params.is_CL:
            raise ValueError(
                "To use CL model, `is_CL` param needs to be set to True, while `has_outliers` - to False"
            )
        self.wf = params.w.astype(dtype=np.float64)
        super().__init__(params, clusters)

    def get_wwt(self) -> np.ndarray:
        ...

    def get_local_edges(self, i: int) -> Set[Tuple[int, int]]:
        ...

    def get_edges(self) -> Set[Tuple[int, int]]:
        edges: Set[Tuple[int, int]] = set()

        for i in self.params.s:
            local_edges = self.get_local_edges(i)
            edges = edges.union(local_edges)

        wwt = self.get_wwt()

        while 2 * len(edges) < self.total_weight:
            a = random.choices(
                population=self.params.w,
                weights=wwt,
                k=utils.randround(self.total_weight / 2) - len(edges),
            )

            b = random.choices(
                population=self.params.w,
                weights=wwt,
                k=utils.randround(self.total_weight / 2) - len(edges),
            )
            for p, q in zip(a, b):
                if p == q:
                    continue
                edges.add(minmax((p, q)))

        return edges


class LocalCLModel(CLModel):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        super().__init__(params, clusters)
        self.xil = np.array(
            [
                self.params.mu / (1 - cl / self.total_weight)
                for cl in self.cluster_weight
            ]
        )
        if self.xil.max() < 1:
            raise ValueError("μ is too large to generate a graph")

    def get_wwt(self) -> np.ndarray:
        return np.array([self.xil[self.clusters[j]] * x for j, x in enumerate(self.wf)])

    def get_local_edges(self, i: int) -> Set[Tuple[int, int]]:
        xi = self.xil[i]
        return _get_local_edges(self.clusters, self.wf, xi, i)


class GlobalCLModel(CLModel):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        super().__init__(params, clusters)
        if not params.xi:
            self.xig = params.mu / (
                1 - (self.cluster_weight**2).sum() / self.total_weight**2
            )
            if self.xig < 1:
                raise ValueError("μ is too large to generate a graph")
        else:
            self.xig = params.xi

    def get_wwt(self) -> np.ndarray:
        return self.xig * self.wf

    def get_local_edges(self, i: int) -> Set[Tuple[int, int]]:
        return _get_local_edges(self.clusters, self.wf, self.xig, i)


def _get_local_edges(clusters, wf, xi, i) -> Set[Tuple[int, int]]:
    local_edges: Set[Tuple[int, int]] = set()
    idx1 = clusters[[clusters == i]]
    w1 = wf[idx1]

    m = utils.randround((1 - xi) * w1.sum() / 2)

    while len(local_edges) < m:
        a = random.choices(population=idx1, weights=w1, k=m - len(local_edges))
        b = random.choices(population=idx1, weights=w1, k=m - len(local_edges))

        for p, q in zip(a, b):
            if p == q:
                continue

            local_edges.add(minmax((p, q)))

    return local_edges
