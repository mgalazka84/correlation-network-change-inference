"""Deterministic envelopes for interval-weight minimum spanning trees.

Statistical coverage is inherited from the input intervals; this module does
not establish or estimate it. Bounds are sharp over the mandatory/possible
edge relaxation, which can be larger than the interval-compatible MST family.
"""
from collections import deque
from itertools import combinations
import numpy as np


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.components = n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.parent[b] = a
        self.components -= 1
        return True


def complete_edges(p):
    return list(combinations(range(p), 2))


def forest(p, edges, ids):
    dsu = DSU(p)
    return all(dsu.union(*edges[e]) for e in sorted(set(ids)))


def components(p, edges, ids):
    dsu = DSU(p)
    for e in ids:
        dsu.union(*edges[e])
        if dsu.components == 1:
            return 1
    return dsu.components


def mst(p, edges, weights):
    dsu = DSU(p)
    selected = set()
    for e in sorted(range(len(edges)), key=lambda i: (weights[i], i)):
        if dsu.union(*edges[e]):
            selected.add(e)
        if len(selected) == p - 1:
            return selected
    raise ValueError("Disconnected graph")


def degrees(p, edges, selected):
    result = np.zeros(p, dtype=int)
    for e in selected:
        a, b = edges[e]
        result[a] += 1
        result[b] += 1
    return result


def interval_envelope(p, edges, lower, upper, tol=1e-12):
    """Return (F,P): every compatible MST contains F and is contained in P.

    Edge e is possible iff e's endpoints cannot be linked by other edges
    with upper bound strictly below lower[e]. It is mandatory iff they
    cannot be linked by other edges with lower bound <= upper[e].
    Positive tol only enlarges P and reduces F, preserving conservatism.
    """
    lower, upper = np.asarray(lower), np.asarray(upper)
    if len(lower) != len(edges) or lower.shape != upper.shape:
        raise ValueError("Invalid dimensions")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Nonfinite intervals")
    if np.any(lower > upper):
        raise ValueError("Empty intervals")
    possible, forced = set(), set()
    for e, (a, b) in enumerate(edges):
        light = DSU(p)
        for f in np.flatnonzero(upper < lower[e] - tol):
            if f != e:
                light.union(*edges[f])
        if light.find(a) != light.find(b):
            possible.add(e)
        alternative = DSU(p)
        for f in np.flatnonzero(lower <= upper[e] + tol):
            if f != e:
                alternative.union(*edges[f])
        if alternative.find(a) != alternative.find(b):
            forced.add(e)
    if not forced <= possible or not forest(p, edges, forced):
        raise AssertionError("Invalid mandatory edge set")
    if components(p, edges, possible) != 1:
        raise ValueError("Input graph is disconnected")
    return forced, possible


def degree_bounds(p, edges, forced, possible):
    """Exact marginal degree ranges over {T tree: F subset T subset P}."""
    dforced = degrees(p, edges, forced)
    cforced = components(p, edges, forced)
    lo, hi = np.zeros(p, int), np.zeros(p, int)
    for v in range(p):
        incident = {e for e in possible if v in edges[e]}
        lo[v] = dforced[v] + components(p, edges, forced | (possible - incident)) - 1
        hi[v] = dforced[v] + cforced - components(p, edges, forced | incident)
    return lo, hi


def maximum_common_independent(ground, independent1, independent2, return_partition=False):
    """Unweighted matroid intersection by shortest augmenting paths.

    Uses standard independence-oracle augmentation. Returns a maximum-cardinality
    set, not a new matroid intersection algorithm.
    """
    ground = set(ground)
    chosen = set()
    # Any common independent set is a valid starting point for augmentation.
    # A greedy warm start avoids repeated scans in dense, weakly constrained
    # envelopes. The unchanged rank certificate still verifies optimality.
    for x in sorted(ground):
        candidate = chosen | {x}
        if independent1(candidate) and independent2(candidate):
            chosen = candidate
    while True:
        outside = sorted(ground - chosen)
        sources = [x for x in outside if independent1(chosen | {x})]
        sinks = {x for x in outside if independent2(chosen | {x})}
        parent = {x: None for x in sources}
        queue = deque(sources)
        endpoint = None
        while queue:
            x = queue.popleft()
            if x in sinks:
                endpoint = x
                break
            if x in chosen:
                successors = [y for y in outside if independent1((chosen - {x}) | {y})]
            else:
                successors = [y for y in sorted(chosen) if independent2((chosen - {y}) | {x})]
            for y in successors:
                if y not in parent:
                    parent[y] = x
                    queue.append(y)
        if endpoint is None:
            # Edmonds min-max certificate: r1(A)+r2(C\A)=|chosen|.
            partition = ground - set(parent)
            return (chosen, partition) if return_partition else chosen
        path = set()
        while endpoint is not None:
            path.add(endpoint)
            endpoint = parent[endpoint]
        chosen ^= path
        if not independent1(chosen) or not independent2(chosen):
            raise AssertionError("Matroid augmentation failed")


def extend_tree(p, edges, forced, possible, common):
    base = set(forced) | set(common)
    dsu = DSU(p)
    for e in sorted(base):
        if not dsu.union(*edges[e]):
            raise ValueError("Cannot extend cyclic set")
    for e in sorted(possible - base):
        if dsu.union(*edges[e]):
            base.add(e)
    if len(base) != p - 1:
        raise ValueError("Cannot extend to a spanning tree")
    return base


def envelope_rank(p, edges, forced, subset):
    return len(forced & subset) + components(p, edges, forced) - components(p, edges, forced | subset)


def turnover_lower_bound(p, edges, env1, env2, witnesses=False, certificate=False):
    """Exact minimum number of replacements over the two relaxed tree families."""
    f1, p1 = env1
    f2, p2 = env2
    common, partition = maximum_common_independent(
        p1 & p2,
        lambda x: forest(p, edges, f1 | x),
        lambda x: forest(p, edges, f2 | x),
        return_partition=True,
    )
    rank_sum = envelope_rank(p,edges,f1,partition) + envelope_rank(p,edges,f2,(p1&p2)-partition)
    assert rank_sum == len(common)
    bound = p - 1 - len(common)
    if certificate:
        return {"turnover_lower":bound,"common_independent":sorted(common),
                "partition_A":sorted(partition),"rank_sum":rank_sum}
    if witnesses:
        t1 = extend_tree(p, edges, f1, p1, common)
        t2 = extend_tree(p, edges, f2, p2, common)
        assert len(t1 - t2) == bound
        return bound, t1, t2
    return bound


def compare_intervals(p, edges, lower1, upper1, lower2, upper2):
    env1 = interval_envelope(p, edges, lower1, upper1)
    env2 = interval_envelope(p, edges, lower2, upper2)
    l1, u1 = degree_bounds(p, edges, *env1)
    l2, u2 = degree_bounds(p, edges, *env2)
    return {
        "turnover_lower": turnover_lower_bound(p, edges, env1, env2),
        "degree_change_lower": l2 - u1,
        "degree_change_upper": u2 - l1,
        "envelope1": env1,
        "envelope2": env2,
        "raw_degree_change_lower": degrees(p, edges, env2[0]) - degrees(p, edges, env1[1]),
        "raw_degree_change_upper": degrees(p, edges, env2[1]) - degrees(p, edges, env1[0]),
    }


def full_envelope_rank(p, edges, forced, possible, subset):
    """Rank of the full-edge matroid whose bases are envelope trees."""
    return envelope_rank(p, edges, forced, set(subset) & possible)


def turnover_upper_bound(p, edges, env1, env2, certificate=False):
    """Exact maximum replacements, by intersection with the second dual.

    The certificate verifies the upper optimum with rank calculations only.
    Ground elements outside possible1 are loops and can be removed.
    """
    f1, p1 = env1
    f2, p2 = env2
    ground_all = set(range(len(edges)))
    rank = p - 1
    def independent1(x):
        return x <= p1 and forest(p, edges, f1 | x)
    def independent_dual2(x):
        return full_envelope_rank(p, edges, f2, p2, ground_all - x) == rank
    chosen, partition = maximum_common_independent(
        p1, independent1, independent_dual2, return_partition=True)
    other = p1 - partition
    dual_rank = len(other) + full_envelope_rank(
        p, edges, f2, p2, ground_all - other) - rank
    rank_sum = full_envelope_rank(p, edges, f1, p1, partition) + dual_rank
    assert rank_sum == len(chosen)
    if certificate:
        return dict(turnover_upper=len(chosen), common_independent=sorted(chosen),
                    partition_A=sorted(partition), ground=sorted(p1), rank_sum=rank_sum)
    return len(chosen)
