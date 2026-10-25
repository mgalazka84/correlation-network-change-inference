"""Independent exhaustive checks against enumerated labelled spanning trees."""
import heapq
import itertools
import json
from pathlib import Path
import numpy as np
from network_bounds import (complete_edges, interval_envelope, degree_bounds,
                           turnover_lower_bound, forest, degrees)


def all_trees(p, edges):
    lookup = {e: i for i, e in enumerate(edges)}
    trees = []
    for seq in itertools.product(range(p), repeat=p-2):
        deg = [1] * p
        for v in seq:
            deg[v] += 1
        leaves = [v for v in range(p) if deg[v] == 1]
        heapq.heapify(leaves)
        t = set()
        for v in seq:
            u = heapq.heappop(leaves)
            t.add(lookup[tuple(sorted((u, v)))])
            deg[u] -= 1
            deg[v] -= 1
            if deg[v] == 1:
                heapq.heappush(leaves, v)
        u, v = leaves
        t.add(lookup[tuple(sorted((u, v)))])
        trees.append(t)
    return trees


def possible_tree(t, trees, lower, upper):
    # A tree is feasible for some box realization iff it is an MST when
    # its own edges are at their lower bounds and all others at their upper.
    w = upper.copy()
    w[list(t)] = lower[list(t)]
    cost = sum(w[list(t)])
    return all(cost <= sum(w[list(s)]) + 1e-10 for s in trees)


def run(seed=701):
    rng = np.random.default_rng(seed)
    counts = {"interval_envelopes": 0, "degree_ranges": 0,
              "turnover_minima": 0, "sound_against_exact_interval_families": 0}
    strict_example = None
    for p, repetitions in [(3, 80), (4, 120), (5, 50)]:
        edges = complete_edges(p)
        trees = all_trees(p, edges)
        tree_degrees = np.array([degrees(p, edges, t) for t in trees])
        for _ in range(repetitions):
            envs, exact_families, relaxed_families, boxes = [], [], [], []
            for s in range(2):
                lower = rng.integers(0, 5, len(edges)).astype(float)
                upper = lower + rng.integers(0, 4, len(edges))
                exact = [t for t in trees if possible_tree(t, trees, lower, upper)]
                f = set.intersection(*exact)
                poss = set.union(*exact)
                actual_env = interval_envelope(p, edges, lower, upper)
                assert actual_env == (f, poss)
                counts["interval_envelopes"] += 1
                indices = [i for i, t in enumerate(trees) if f <= t <= poss]
                relaxed = [trees[i] for i in indices]
                lo, hi = degree_bounds(p, edges, f, poss)
                assert np.array_equal(lo, tree_degrees[indices].min(0))
                assert np.array_equal(hi, tree_degrees[indices].max(0))
                counts["degree_ranges"] += 2 * p
                envs.append(actual_env)
                boxes.append({"lower":lower.tolist(),"upper":upper.tolist()})
                exact_families.append(exact)
                relaxed_families.append(relaxed)
            actual = min(len(a-b) for a in relaxed_families[0] for b in relaxed_families[1])
            predicted, a, b = turnover_lower_bound(p, edges, *envs, witnesses=True)
            assert predicted == actual and len(a-b) == predicted
            counts["turnover_minima"] += 1
            exact_actual = min(len(a-b) for a in exact_families[0] for b in exact_families[1])
            assert predicted <= exact_actual
            counts["sound_against_exact_interval_families"] += 1
            if exact_actual > predicted and strict_example is None:
                strict_example = {"p":p, "relaxed_turnover":predicted,
                                  "exact_box_turnover":exact_actual,
                                  "boxes":boxes,
                                  "forced1":sorted(envs[0][0]), "possible1":sorted(envs[0][1]),
                                  "forced2":sorted(envs[1][0]), "possible2":sorted(envs[1][1])}
    # Point intervals with exact ties must retain all spanning trees.
    p = 5
    edges = complete_edges(p)
    f, poss = interval_envelope(p, edges, np.ones(10), np.ones(10))
    assert not f and poss == set(range(10))
    lo, hi = degree_bounds(p, edges, f, poss)
    assert np.all(lo == 1) and np.all(hi == p-1)
    assert turnover_lower_bound(p, edges, (f, poss), (f, poss)) == 0
    result = {"status":"passed", "seed":seed, "counts":counts,
              "exact_tie_check":"passed", "relaxation_gap_example":strict_example,
              "note":"Exhaustive checks support code correctness, not statistical coverage or novelty."}
    root = Path(__file__).resolve().parents[1]
    (root / "validation" / "exhaustive_checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
