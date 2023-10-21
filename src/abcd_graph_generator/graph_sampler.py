__all__ = [
    "gen_graph",
]

import collections
import copy
from typing import (
    NamedTuple,
    Set,
    Tuple,
)

import numpy as np

import abcd_graph_generator.utils as utils
from abcd_graph_generator.abcd_params import ABCDParams
from abcd_graph_generator.model.cl_model import cl_model
from abcd_graph_generator.model.config_model import config_model


def populate_clusters(params: ABCDParams) -> np.ndarray:
    tabu_set: Set[int] = set()

    mul = utils.get_mul(params)

    slots = copy.copy(params.s)
    clusters = np.ones(len(params.w)) * (-1)

    if params.has_outliers:
        tabu = utils.get_tabu(params)
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

        loc = np.random.choice(range(j0 + 1, j + 1), wts)
        clusters[i] = loc
        slots[loc] -= 1

    _raise_for_values(slots, clusters)

    return clusters


def _raise_for_values(slots: np.ndarray, clusters: np.ndarray) -> None:
    slots_sum = slots.sum()
    if slots_sum != 0:
        raise ValueError(f"Slots sum expected to be 0. Actual: {slots_sum}.")

    min_clusters = np.min(clusters)
    if min_clusters != 1:
        raise ValueError(f"Cluster min expected to be 1. Actual: {min_clusters}.")


def gen_graph(params: ABCDParams) -> NamedTuple[Set[Tuple[int, int]], np.ndarray]:
    result = collections.namedtuple("result", ["edges", "clusters"])
    clusters = populate_clusters(params)

    edges = cl_model(clusters, params) if params.is_CL else config_model(clusters, params)
    return result(edges=edges, clusters=clusters)
