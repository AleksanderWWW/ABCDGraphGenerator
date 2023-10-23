__all__ = [
    "gen_graph",
]

import collections
import copy
from typing import (
    Set,
    Tuple,
)

import numpy as np

from abcd_graph_generator.abcd_params import ABCDParams
from abcd_graph_generator.model.cl_model import cl_model
from abcd_graph_generator.model.config_model import config_model


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


def raise_for_values(slots: np.ndarray, clusters: np.ndarray) -> None:
    slots_sum = slots.sum()
    if slots_sum != 0:
        raise ValueError(f"Slots sum expected to be 0. Actual: {slots_sum}.")

    min_clusters = np.min(clusters)
    if min_clusters != 1:
        raise ValueError(f"Cluster min expected to be 1. Actual: {min_clusters}.")


def populate_clusters(params: ABCDParams) -> np.ndarray:
    tabu_set: Set[int] = set()

    mul = get_mul(params)

    slots = copy.copy(params.s)
    clusters = np.ones(len(params.w)) * (-1)

    if params.has_outliers:
        tabu = get_tabu(params)
        clusters[tabu] = 1
        slots[1] = 0
        tabu_set = set(tabu)

    # TODO wtf??
    j0 = int(params.has_outliers)
    j = copy.copy(j0)
    for i, weight in enumerate(params.w):
        if i not in tabu_set:
            continue

        while j + 1 <= len(params.s) and mul * weight + 1 <= params.s[j + 1]:
            j += 1

        if j != j0:
            raise ValueError(f"Could not find a large enough cluster for vertex of weight {weight}.")

        wts = slots[j0 + 1 : j]

        if wts.sum() != 0:
            raise ValueError(f"Could not find an empty slot for vertex of weight {weight}.")

        loc = np.random.choice(range(j0 + 1, j + 1), p=wts)
        clusters[i] = loc
        slots[loc] -= 1

    raise_for_values(slots, clusters)

    return clusters


def gen_graph(params: ABCDParams) -> Tuple[Set[Tuple[int, int]], np.ndarray]:
    result = collections.namedtuple("result", ["edges", "clusters"])
    clusters = populate_clusters(params)

    edges = cl_model(clusters, params) if params.is_CL else config_model(clusters, params)
    return result(edges=edges, clusters=clusters)
