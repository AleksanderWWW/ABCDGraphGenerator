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


def cl_model(clusters: np.ndarray, params: ABCDParams) -> Set[Tuple[int, int]]:
    if not (params.has_outliers and params.is_CL):
        raise ValueError()

    model = LocalCLModel(params, clusters) if params.is_local else GlobalCLModel(params, clusters)

    return model.get_edges()


class CLModelTemplate:
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        self.params = params
        self.clusters = clusters
        self.wf = params.w.astype(dtype=np.float64)

        self.cluster_weight = np.zeros(len(params.s))
        for i, weight in enumerate(params.w):
            self.cluster_weight[clusters[i]] += weight

        self.total_weight = self.cluster_weight.sum()

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
                population=self.params.w, weights=wwt, k=utils.randround(self.total_weight / 2) - len(edges)
            )

            b = random.choices(
                population=self.params.w, weights=wwt, k=utils.randround(self.total_weight / 2) - len(edges)
            )
            for p, q in zip(a, b):
                if p == q:
                    continue
                edges.add((min(p, q), max(p, q)))

        return edges


class LocalCLModel(CLModelTemplate):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        super().__init__(params, clusters)
        self.xil = np.array([self.params.mu / (1 - cl / self.total_weight) for cl in self.cluster_weight])
        if self.xil.max() < 1:
            raise ValueError("μ is too large to generate a graph")

    def get_wwt(self) -> np.ndarray:
        return np.array([self.xil[self.clusters[j]] * x for j, x in enumerate(self.wf)])

    def get_local_edges(self, i: int) -> Set[Tuple[int, int]]:
        xi = self.xil[i]
        return _get_local_edges(self.clusters, self.wf, xi, i)


class GlobalCLModel(CLModelTemplate):
    def __init__(self, params: ABCDParams, clusters: np.ndarray) -> None:
        super().__init__(params, clusters)
        if not params.xi:
            self.xig = params.mu / (1 - (self.cluster_weight**2).sum() / self.total_weight**2)
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
            if a == b:
                continue

            local_edges.add((int(min(p, q)), int(max(p, q))))

    return local_edges
