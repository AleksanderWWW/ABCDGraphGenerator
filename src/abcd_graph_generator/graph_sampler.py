import copy
import random
from typing import Set

import numpy as np

import abcd_graph_generator.utils as utils
from abcd_graph_generator.abcd_params import ABCDParams


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


def cl_model(clusters: np.ndarray, params: ABCDParams) -> Set:
    xil, xig = np.array([]), None
    if not (params.has_outliers and params.is_CL):
        raise ValueError()

    cluster_weight = np.zeros(len(params.s))
    for i, weight in enumerate(params.w):
        cluster_weight[clusters[i]] += weight

    total_weight = cluster_weight.sum()
    if params.is_local:
        xil = np.array([params.mu / (1 - cl / total_weight) for cl in cluster_weight])
        if xil.max() < 1:
            raise ValueError("μ is too large to generate a graph")
    else:
        if not params.xi:
            xig = params.mu / (1 - (cluster_weight**2).sum() / total_weight**2)
            if xig < 1:
                raise ValueError("μ is too large to generate a graph")
        else:
            xig = params.xi

    wf = params.w.astype(dtype=np.float64)
    edges = set()

    for i in params.s:
        local_edges = set()
        idx1 = clusters[[clusters == i]]
        w1 = wf[idx1]

        xi = xil[i] if params.is_local else xig
        m = utils.randround((1 - xi) * w1.sum() / 2)

        while len(local_edges) < m:
            a = random.choices(population=idx1, weights=w1, k=m - len(local_edges))
            b = random.choices(population=idx1, weights=w1, k=m - len(local_edges))

            for p, q in zip(a, b):
                if a == b:
                    continue

                local_edges.add((min(p, q), max(p, q)))
        edges = edges.union(local_edges)

    wwt = np.array([xil[clusters[j]] * x for j, x in enumerate(wf)]) if params.is_local else xig * wf

    while 2 * len(edges) < total_weight:
        a = random.choices(population=params.w, weights=wwt, k=utils.randround(total_weight / 2) - len(edges))
        b = random.choices(population=params.w, weights=wwt, k=utils.randround(total_weight / 2) - len(edges))
        for p, q in zip(a, b):
            if p == q:
                continue
            edges.add((min(p, q), max(p, q)))

    return edges
