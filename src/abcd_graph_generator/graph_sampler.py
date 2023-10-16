import copy
from typing import Set

import numpy as np

from abcd_graph_generator.abcd_params import ABCDParams
from abcd_graph_generator.utils import get_mul


def populate_clusters(params: ABCDParams) -> np.ndarray:
    mul = get_mul(params)

    slots = copy.deepcopy(params.s)
    clusters = np.ones(len(params.w)) * (-1)

    if params.has_outliers:
        nout = params.s[0]
        n = len(params.w)
        L = sum(map(lambda d: min(1.0, params.xi * d), params.w))
        threshold = L + nout - L * nout / n - 1.0
        idx = np.searchsorted(params.w, threshold, side="right")  # TODO to be confirmed

        if len(params.w[idx:]) < nout:
            raise ValueError("not enough nodes feasible for classification as outliers")

        tabu = np.random.choice(range(idx, n), nout, replace=False)  # TODO n or n+1?
        clusters[tabu] = 1
        slots[1] = 0
        stabu = set(tabu)

    else:
        stabu: Set[int] = set()

    # TODO wtf??
    j0 = int(params.has_outliers)
    j = j0
    for (i, vw) in enumerate(params.w):
        if i not in stabu:
            continue

        while j + 1 <= len(params.s) and mul * vw + 1 <= params.s[j + 1]:
            j += 1

        if j != j0:
            raise Exception(f"could not find a large enough cluster for vertex of weight {vw}")

        wts = np.array(slots[j0 + 1 : j])

        if wts.sum() != 0:
            raise Exception(f"could not find an empty slot for vertex of weight {vw}")

        loc = np.random.choice(range(j0 + 1, j), wts)
        clusters[i] = loc
        slots[loc] -= 1

    assert sum(slots) == 0
    assert min(clusters) == 1
    return clusters
