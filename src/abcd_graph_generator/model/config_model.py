__all__ = [
    "config_model",
]

from typing import (
    Set,
    Tuple,
)

import numpy as np

from abcd_graph_generator import ABCDParams
from abcd_graph_generator.model.model import GraphGenModel


class ConfigModel(GraphGenModel):
    def __init__(self, clusters: np.ndarray, params: ABCDParams) -> None:
        if params.is_CL or params.is_local:
            raise ValueError("Neither `is_CL` nor `is_local` param can be set to True")
        super().__init__(params, clusters)

    def _get_xig(self) -> float:
        if self.params.has_outliers:
            raise ValueError
        _xig = self.params.mu / (1.0 - np.square(self.cluster_weight).sum() / self.total_weight**2)
        if _xig >= 1:
            raise ValueError("μ is too large to generate a graph")
        return _xig

    def get_edges(self) -> Set[Tuple[int, int]]:
        xig = self.params.xi if self.params.xi else self._get_xig()

        w_internal_raw = np.array([self.params.w[i] * (1 - xig) for i in self.params.w])

        if self.params.has_outliers:
            for i in self.clusters[self.clusters == 1]:
                w_internal_raw[i] = 0


def config_model(clusters: np.ndarray, params: ABCDParams) -> Set[Tuple[int, int]]:
    if params.is_CL or params.is_local:
        raise ValueError

    cluster_weight = np.zeros(len(params.s)).astype(dtype=int)
    for i, weight in enumerate(params.w):
        cluster_weight[clusters[i]] += weight

    total_weight = cluster_weight.sum()

    def _get_xig() -> float:
        if params.has_outliers:
            raise ValueError
        _xig = params.mu / (1.0 - np.square(cluster_weight).sum() / total_weight**2)
        if _xig >= 1:
            raise ValueError("μ is too large to generate a graph")
        return _xig

    if params.is_local:
        xil = np.array([params.mu / (1.0 - clw / total_weight) for clw in cluster_weight])
        if xil.max() >= 1:
            raise ValueError("μ is too large to generate a graph")

        w_internal_raw = np.array([params.w[i] * (1 - xil[clusters[i]]) for i in params.w])
    else:
        xig = params.xi if params.xi else _get_xig()

        w_internal_raw = np.array([params.w[i] * (1 - xig) for i in params.w])

        if params.has_outliers:
            for i in clusters[clusters == 1]:
                w_internal_raw[i] = 0

    # clusterlist = [Int[] for i in axes(s, 1)]
    # for i in axes(clusters, 1)
    #     push!(clusterlist[clusters[i]], i)
    # end
    #
    # edges = Set
    # {Tuple
    # {Int, Int}}()
    #
    # unresolved_collisions = 0
    # w_internal = zeros(Int, length(w_internal_raw))
    # for cluster in clusterlist
    #     maxw_idx = argmax(view(w_internal_raw, cluster))
    #     wsum = 0
    #     for i in axes(cluster, 1)
    #         if i != maxw_idx
    #             neww = randround(w_internal_raw[cluster[i]])
    #             w_internal[cluster[i]] = neww
    #             wsum += neww
    #         end
    #     end
    #     maxw = floor(Int, w_internal_raw[cluster[maxw_idx]])
    #     w_internal[cluster[maxw_idx]] = maxw + (isodd(wsum) ? iseven(maxw): isodd(maxw))
    #     if w_internal[cluster[maxw_idx]] > w[cluster[maxw_idx]]
    #         @
    #
    #
    #         assert w[cluster[maxw_idx]] + 1 == w_internal[cluster[maxw_idx]]
    #         w[cluster[maxw_idx]] += 1
    #     end
    #
    #     if params.hasoutliers & & cluster === clusterlist[1]
    #
    #
    #     @
    #
    #
    #     assert findall(clusters. == 1) == cluster
    #
    #
    #     @
    #
    #
    #     assert all(iszero, w_internal[cluster])
    # end
    # stubs = Int[]
    # for i in cluster
    #     for j in 1: w_internal[i]
    #     push!(stubs, i)
    # end
    # end
    #
    #
    # @
    #
    #
    # assert sum(w_internal[cluster]) == length(stubs)
    #
    #
    # @
    #
    #
    # assert iseven(length(stubs))
    # if params.hasoutliers & & cluster === clusterlist[1]
    #
    #
    # @
    #
    #
    # assert isempty(stubs)
    # end
    # shuffle!(stubs)
    # local_edges = Set
    # {Tuple
    # {Int, Int}}()
    # recycle = Tuple
    # {Int, Int}[]
    # for i in 1: 2: length(stubs)
    # e = minmax(stubs[i], stubs[i + 1])
    # if (e[1] == e[2]) | | (e in local_edges)
    #     push!(recycle, e)
    # else
    #     push!(local_edges, e)
    # end
    # end
    # last_recycle = length(recycle)
    # recycle_counter = last_recycle
    # while !isempty(recycle)
    # recycle_counter -= 1
    # if recycle_counter < 0
    #     if length(recycle) < last_recycle
    #         last_recycle = length(recycle)
    #         recycle_counter = last_recycle
    #     else
    #         break
    #     end
    # end
    # p1 = popfirst!(recycle)
    # from_recycle = 2 * length(recycle) / length(stubs)
    # success = false
    # for _ in 1: 2: length(stubs)
    # p2 =
    # if rand() < from_recycle
    #     used_recycle = true
    #     recycle_idx = rand(axes(recycle, 1))
    #     recycle[recycle_idx]
    # else
    #     used_recycle = false
    #     rand(local_edges)
    # end
    # if rand() < 0.5
    #     newp1 = minmax(p1[1], p2[1])
    #     newp2 = minmax(p1[2], p2[2])
    # else
    #     newp1 = minmax(p1[1], p2[2])
    #     newp2 = minmax(p1[2], p2[1])
    # end
    # if newp1 == newp2
    #     good_choice = false
    # elseif(newp1[1] == newp1[2]) | | (newp1 in local_edges)
    # good_choice = false
    # elseif(newp2[1] == newp2[2]) | | (newp2 in local_edges)
    # good_choice = false
    # else
    # good_choice = true
    # end
    # if good_choice
    #     if used_recycle
    #         recycle[recycle_idx], recycle[end] = recycle[end], recycle[recycle_idx]
    #         pop!(recycle)
    #     else
    #         pop!(local_edges, p2)
    #     end
    #     success = true
    #     push!(local_edges, newp1)
    #     push!(local_edges, newp2)
    #     break
    # end
    # end
    # success | | push!(recycle, p1)
    # end
    # old_len = length(edges)
    # union!(edges, local_edges)
    #
    #
    # @
    #
    #
    # assert length(edges) == old_len + length(local_edges)
    #
    #
    # @
    #
    #
    # assert 2 * (length(local_edges) + length(recycle)) == length(stubs)
    # for (a, b) in recycle
    #     w_internal[a] -= 1
    #     w_internal[b] -= 1
    # end
    # unresolved_collisions += length(recycle)
    # end
    #
    # if unresolved_collisions > 0
    #     println("Unresolved_collisions: ", unresolved_collisions,
    #             "; fraction: ", 2 * unresolved_collisions / total_weight)
    # end
    #
    # stubs = Int[]
    # for i in axes(w, 1)
    #     for j in w_internal[i] + 1: w[i]
    #     push!(stubs, i)
    # end
    # end
    #
    #
    # @
    #
    #
    # assert sum(w) == length(stubs) + sum(w_internal)
    # if params.hasoutliers
    #     if 2 * sum(w[clusters. == 1]) > length(stubs)
    #         @warn
    #
    #
    #         "Because of low value of ξ the outlier nodes form a community. " *
    #         "It is recommended to increase ξ."
    # end
    # end
    # shuffle!(stubs)
    # if isodd(length(stubs))
    #     maxi = 1
    #
    #
    #     @
    #
    #
    #     assert w[stubs[maxi]] > w_internal[stubs[maxi]]
    #     for i in 2: length(stubs)
    #     si = stubs[i]
    #
    #
    #     @
    #
    #
    #     assert w[si] > w_internal[si]
    #     if w[si] > w[stubs[maxi]]
    #         maxi = i
    #     end
    # end
    # si = popat!(stubs, maxi)
    #
    #
    # @
    #
    #
    # assert w[si] > w_internal[si]
    # w[si] -= 1
    # end
    # global_edges = Set
    # {Tuple
    # {Int, Int}}()
    # recycle = Tuple
    # {Int, Int}[]
    # for i in 1: 2: length(stubs)
    # e = minmax(stubs[i], stubs[i + 1])
    # if (e[1] == e[2]) | | (e in global_edges) | | (e in edges)
    #     push!(recycle, e)
    # else
    #     push!(global_edges, e)
    # end
    # end
    # last_recycle = length(recycle)
    # recycle_counter = last_recycle
    # while !isempty(recycle)
    # recycle_counter -= 1
    # if recycle_counter < 0
    #     if length(recycle) < last_recycle
    #         last_recycle = length(recycle)
    #         recycle_counter = last_recycle
    #     else
    #         break
    #     end
    # end
    # p1 = pop!(recycle)
    # from_recycle = 2 * length(recycle) / length(stubs)
    # p2 =
    # if rand() < from_recycle
    #     i = rand(axes(recycle, 1))
    #     recycle[i], recycle[end] = recycle[end], recycle[i]
    #     pop!(recycle)
    # else
    #     x = rand(global_edges)
    #     pop!(global_edges, x)
    # end
    # if rand() < 0.5
    #     newp1 = minmax(p1[1], p2[1])
    #     newp2 = minmax(p1[2], p2[2])
    # else
    #     newp1 = minmax(p1[1], p2[2])
    #     newp2 = minmax(p1[2], p2[1])
    # end
    # for newp in (newp1, newp2)
    #     if (newp[1] == newp[2]) | | (newp in global_edges) | | (newp in edges)
    #         push!(recycle, newp)
    #     else
    #         push!(global_edges, newp)
    #     end
    # end
    # end
    # old_len = length(edges)
    # union!(edges, global_edges)
    #
    #
    # @
    #
    #
    # assert length(edges) == old_len + length(global_edges)
    # if isempty(recycle)
    #     @
    #
    #
    #     assert 2 * length(global_edges) == length(stubs)
    # else
    #     last_recycle = length(recycle)
    #     recycle_counter = last_recycle
    #     while !isempty(recycle)
    #     recycle_counter -= 1
    #     if recycle_counter < 0
    #         if length(recycle) < last_recycle
    #             last_recycle = length(recycle)
    #             recycle_counter = last_recycle
    #         else
    #             break
    #         end
    #     end
    #     p1 = pop!(recycle)
    #     x = rand(edges)
    #     p2 = pop!(edges, x)
    #     if rand() < 0.5
    #         newp1 = minmax(p1[1], p2[1])
    #         newp2 = minmax(p1[2], p2[2])
    #     else
    #         newp1 = minmax(p1[1], p2[2])
    #         newp2 = minmax(p1[2], p2[1])
    #     end
    #     for newp in (newp1, newp2)
    #         if (newp[1] == newp[2]) | | (newp in edges)
    #             push!(recycle, newp)
    #         else
    #             push!(edges, newp)
    #         end
    #     end
    # end
    # end
    # if !isempty(recycle)
    # unresolved_collisions = length(recycle)
    # println("Very hard graph. Failed to generate ", unresolved_collisions,
    #         "edges; fraction: ", 2 * unresolved_collisions / total_weight)
    # end
    # return edges
    # end
