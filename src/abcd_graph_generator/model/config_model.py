__all__ = [
    "config_model",
]

import random
from typing import (
    List,
    Set,
    Tuple,
    Optional,
)

import numpy as np

from abcd_graph_generator import (
    ABCDParams,
    utils,
)
from abcd_graph_generator.model.model import GraphGenModel


def get_stubs(
    w_internal: np.ndarray, cluster: List[int], params: ABCDParams, idx: int
) -> np.ndarray:
    stubs = []
    for i in cluster:
        for j in range(w_internal[i] + 1):
            stubs.append(i)
    stubs = np.array(stubs, dtype=int)

    if w_internal[cluster].sum() != len(stubs) or utils.is_odd(len(stubs)):
        raise RuntimeError

    if idx == 0 and params.has_outliers:
        if len(stubs) != 0:
            raise RuntimeError

    np.random.shuffle(stubs)

    return stubs


class ConfigModel(GraphGenModel):
    def __init__(self, clusters: np.ndarray, params: ABCDParams) -> None:
        self.cluster_list: Optional[List[List[int]]] = None
        self.w_internal_raw: Optional[np.ndarray[float]] = None
        if params.is_CL or params.is_local:
            raise ValueError(
                "Neither `is_CL` nor `is_local` param can be set to True in config model"
            )
        super().__init__(params, clusters)

    def setup(self) -> None:
        xig = self.params.xi if self.params.xi else self._get_xig()

        self.w_internal_raw = np.array(
            [self.params.w[i] * (1 - xig) for i in self.params.w]
        )

        if self.params.has_outliers:
            for i in self.clusters[self.clusters == 1]:
                self.w_internal_raw[i] = 0

        self.cluster_list = [[] for _ in self.params.s]
        for i in self.clusters:
            self.cluster_list[self.clusters[i]].append(i)

    def _get_xig(self) -> float:
        if self.params.has_outliers:
            raise ValueError
        _xig = self.params.mu / (
            1.0 - np.square(self.cluster_weight).sum() / self.total_weight**2
        )
        if _xig >= 1:
            raise ValueError("μ is too large to generate a graph")
        return _xig

    def get_edges(self) -> Set[Tuple[int, int]]:
        self.setup()

        edges: Set[Tuple[int, int]] = set()

        unresolved_collisions = 0
        w_internal = np.zeros(len(self.w_internal_raw), dtype=int)

        for idx, cluster in enumerate(self.cluster_list):
            max_w_idx = np.argmax(self.w_internal_raw[cluster])
            w_sum = 0
            for i in cluster:
                if i == max_w_idx:
                    continue

                new_w = utils.randround(self.w_internal_raw[cluster[i]])

                w_internal[cluster[i]] = new_w
                w_sum += new_w

            max_w = np.floor(self.w_internal_raw[cluster[max_w_idx]])
            w_internal[cluster[max_w_idx]] = max_w + (
                utils.is_odd(w_sum) if utils.is_even(max_w) else utils.is_odd(max_w)
            )

            if w_internal[cluster[max_w_idx]] <= self.params.w[cluster[max_w_idx]]:
                continue

            if self.params.w[cluster[max_w_idx]] + 1 != w_internal[cluster[max_w_idx]]:
                raise ValueError

            self.params.w[cluster[max_w_idx]] += 1

            if not (self.params.has_outliers and cluster == self.cluster_list[0]):
                continue

            if not np.array_equal(self.clusters[self.clusters == 1], np.array(cluster)):
                raise ValueError

            if not np.all(w_internal[cluster] == 0):
                raise ValueError

            stubs = get_stubs(w_internal, cluster, self.params, idx)

            local_edges: Set[Tuple[int, int]] = set()
            recycle: List[Tuple[int, int]] = []

            for i in range(0, len(stubs), 2):
                edge = (
                    np.min([stubs[i], stubs[i + 1]]),
                    np.max([stubs[i], stubs[i + 1]]),
                )
                if edge[0] == edge[1] or edge in local_edges:
                    recycle.append(edge)
                else:
                    local_edges.add(edge)

            last_recycle = len(recycle)
            recycle_counter = last_recycle
            while recycle:
                recycle_counter -= 1
                if recycle_counter < 0:
                    if len(recycle) < last_recycle:
                        last_recycle = len(recycle)
                        recycle_counter = last_recycle
                    else:
                        break

                p1 = recycle.pop(0)
                from_recycle = 2 * len(recycle) / len(stubs)
                success = False

                for _ in range(0, len(stubs), 2):
                    if random.random() < from_recycle:
                        used_recycle = True
                        recycle_idx = random.randint(0, len(recycle) - 1)
                        p2 = recycle.pop(recycle_idx)
                    else:
                        used_recycle = False
                        p2 = random.choice(list(local_edges))

                    if random.random() < 0.5:
                        newp1 = (min(p1[0], p2[0]), max(p1[1], p2[1]))
                        newp2 = (min(p1[0], p2[1]), max(p1[1], p2[0]))
                    else:
                        newp1 = (min(p1[0], p2[1]), max(p1[1], p2[0]))
                        newp2 = (min(p1[0], p2[0]), max(p1[1], p2[1]))

                    good_choice = True

                    if newp1 == newp2 or newp1[0] == newp1[1] or newp1 in local_edges:
                        good_choice = False
                    elif newp2[0] == newp2[1] or newp2 in local_edges:
                        good_choice = False

                    if good_choice:
                        if used_recycle:
                            success = True
                            local_edges.remove(p2)
                        local_edges.add(newp1)
                        local_edges.add(newp2)
                        break

                if not success:
                    recycle.append(p1)

            old_len = len(edges)
            edges.update(local_edges)
            assert len(edges) == old_len + len(local_edges)
            assert 2 * (len(local_edges) + len(recycle)) == len(stubs)

            for a, b in recycle:
                w_internal[a] -= 1
                w_internal[b] -= 1

            unresolved_collisions += len(recycle)

        if unresolved_collisions > 0:
            print(
                f"Unresolved_collisions: {unresolved_collisions}; fraction: {2 * unresolved_collisions / self.total_weight}"
            )

        stubs = []
        for i in range(len(self.params.w)):
            for j in range(w_internal[i] + 1, self.params.w[i]):
                stubs.append(i)

        assert sum(self.params.w) == len(stubs) + sum(w_internal)

        if self.params.has_outliers:
            if 2 * sum(
                [
                    self.params.w[i]
                    for i, cluster in enumerate(self.clusters)
                    if cluster == 1
                ]
            ) > len(stubs):
                print(
                    "Because of low value of ξ the outlier nodes form a community. It is recommended to increase ξ."
                )

        random.shuffle(stubs)

        if len(stubs) % 2 == 1:
            maxi = 0
            assert self.params.w[stubs[maxi]] > w_internal[stubs[maxi]]
            for i in range(1, len(stubs)):
                si = stubs[i]
                assert self.params.w[si] > w_internal[si]
                if self.params.w[si] > self.params.w[stubs[maxi]]:
                    maxi = i
            si = stubs.pop(maxi)
            assert self.params.w[si] > w_internal[si]
            self.params.w[si] -= 1

        global_edges = set()
        recycle = []
        for i in range(0, len(stubs), 2):
            e = (min(stubs[i], stubs[i + 1]), max(stubs[i], stubs[i + 1]))
            if e[0] == e[1] or e in global_edges or e in edges:
                recycle.append(e)
            else:
                global_edges.add(e)

        last_recycle = len(recycle)
        recycle_counter = last_recycle

        while recycle:
            recycle_counter -= 1
            if recycle_counter < 0:
                if len(recycle) < last_recycle:
                    last_recycle = len(recycle)
                else:
                    break

            p1 = recycle.pop(0)
            from_recycle = 2 * len(recycle) / len(stubs)

            if random.random() < from_recycle:
                i = random.randint(0, len(recycle) - 1)
                recycle[i], recycle[-1] = recycle[-1], recycle[i]
                recycle.pop()
                p2 = recycle.pop(i)
            else:
                x = random.choice(list(global_edges))
                global_edges.remove(x)
                p2 = x

            if random.random() < 0.5:
                newp1 = (min(p1[0], p2[0]), max(p1[1], p2[1]))
                newp2 = (min(p1[1], p2[1]), max(p1[0], p2[0]))
            else:
                newp1 = (min(p1[0], p2[1]), max(p1[1], p2[0]))
                newp2 = (min(p1[1], p2[0]), max(p1[0], p2[1]))

            for newp in (newp1, newp2):
                if newp[0] == newp[1] or newp in edges:
                    recycle.append(newp)
                else:
                    edges.add(newp)

                old_len = len(edges)
                edges.update(global_edges)
                assert len(edges) == old_len + len(global_edges)

                if not recycle:
                    assert 2 * len(global_edges) == len(stubs)
                else:
                    last_recycle = len(recycle)
                recycle_counter = last_recycle

            while recycle:
                recycle_counter -= 1
                if recycle_counter < 0:
                    if len(recycle) < last_recycle:
                        last_recycle = len(recycle)
                        recycle_counter = last_recycle
                    else:
                        break

                p1 = recycle.pop(0)
                x = random.choice(list(edges))
                edges.discard(x)  # Use discard to remove x from the set

                if random.random() < 0.5:
                    newp1 = (min(p1[0], p2[0]), max(p1[1], p2[1]))
                    newp2 = (min(p1[1], p2[1]), max(p1[0], p2[0]))
                else:
                    newp1 = (min(p1[0], p2[1]), max(p1[1], p2[0]))
                    newp2 = (min(p1[1], p2[0]), max(p1[0], p2[1]))

                for newp in (newp1, newp2):
                    if newp[0] == newp[1] or newp in edges:
                        recycle.append(newp)
                    else:
                        edges.add(newp)

                    if recycle:
                        unresolved_collisions = len(recycle)
                    print(
                        f"Very hard graph. Failed to generate {unresolved_collisions} edges; fraction: {2 * unresolved_collisions / self.total_weight}"
                    )

        return edges


def config_model(clusters: np.ndarray, params: ABCDParams) -> Set[Tuple[int, int]]:
    return ConfigModel(clusters, params).get_edges()
