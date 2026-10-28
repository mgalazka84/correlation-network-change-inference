"""Pre-specified revision studies, with complete paired replication output.

The protocol was fixed after the referee report and original study, before
the new simulation outputs. It is not a preregistration or confirmatory trial.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse, heapq, itertools, json, time, zipfile, platform, sys, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from paired_order import fundamental_comparisons
from finite_calibration import kendall_tau_a
from network_bounds import complete_edges, mst, DSU
from legacy_comparators import compare_legacy
from run_study import box_compare, order_error, covariance, design as original_design
from revision_methods import make_plans, from_plans, pilot_evidence, multiplier_compare, kernel_projection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'revision_results'
MASTER = 281028601


def random_tree(p, seed):
    rng = np.random.default_rng(seed); code = rng.integers(0, p, p-2)
    degree = np.ones(p, int)+np.bincount(code, minlength=p)
    heap = list(np.flatnonzero(degree == 1)); heapq.heapify(heap); result = []
    for v in code:
        leaf = heapq.heappop(heap); result.append(tuple(sorted((int(v), int(leaf)))))
        degree[v] -= 1
        if degree[v] == 1: heapq.heappush(heap, int(v))
    result.append(tuple(sorted(map(int, heap))))
    return sorted(result)


def changed_tree(tree, p, k, seed):
    original = set(map(tuple, tree)); current = set(original)
    rng = np.random.default_rng(seed)
    for _ in range(k):
        eligible = sorted(current & original)
        rng.shuffle(eligible)
        for removed in eligible:
            d = DSU(p)
            for a, b in current-{removed}: d.union(a, b)
            candidates = [e for e in complete_edges(p) if e not in original and
                          e not in current and d.find(e[0]) != d.find(e[1])]
            if candidates:
                added = candidates[int(rng.integers(len(candidates)))]; break
        else: raise AssertionError('Cannot perform requested edge changes')
        current.remove(removed); current.add(added)
    assert len(original-current) == k
    return sorted(current)


def tree_covariance(tree, p, rho):
    adjacency = [[] for _ in range(p)]
    for a, b in tree: adjacency[a].append(b); adjacency[b].append(a)
    R = np.eye(p)
    for root in range(p):
        todo = [(root, -1, 1.)]
        while todo:
            node, parent, value = todo.pop(); R[root, node] = value
            todo += [(v, node, value*rho) for v in adjacency[node] if v != parent]
    np.linalg.cholesky(R)
    return R


def wine_topology():
    with zipfile.ZipFile(ROOT/'data/wine_quality.zip') as z:
        frame = pd.read_csv(z.open('winequality-red.csv'), sep=';').drop(columns='quality')
    edges = complete_edges(11); tree = mst(11, edges, -kendall_tau_a(frame.to_numpy()))
    return [edges[i] for i in sorted(tree)]


def revision_design():
    cells = []
    def add(shape, p, n=1000, rho=.65, changes=2, model='Gaussian', epsilon=None):
        if shape == 'star':
            t1 = [(0, j) for j in range(1, p)]
            t2 = sorted(tuple(sorted((1, j))) for j in range(p) if j != 1) if changes else t1
            if changes: changes = p-2
        elif shape in ('path', 'random', 'wine-topology'):
            t1 = ([(i, i+1) for i in range(p-1)] if shape == 'path' else
                  random_tree(p, 72103+p) if shape == 'random' else wine_topology())
            t2 = changed_tree(t1, p, changes, 94307+p+changes) if changes else t1
        else: t1 = t2 = []
        cells.append(dict(cell=len(cells), shape=shape, p=p, n=n, rho=rho,
                          changes=changes, model=model, epsilon=epsilon,
                          tree1=t1, tree2=t2))
    for shape, p, changes in itertools.product(['path', 'random'], [8, 16], [0, 1, 2]):
        add(shape, p, changes=changes)
    add('star', 8, rho=.35); add('star', 8); add('star', 8, model='Binary-Gaussian')
    add('star', 16)
    for shape, changes in itertools.product(['star', 'path', 'random'], [0, 2]):
        add(shape, 32, changes=changes)
    for changes in [0, 2]: add('wine-topology', 11, changes=changes)
    for p, changes in itertools.product([8, 16], [0, 1]): add('path', p, rho=.98, changes=changes)
    for shape, p in itertools.product(['equicorrelated', 'near-equal'], [8, 16]):
        add(shape, p, rho=.35, changes=0, epsilon=.005 if shape == 'near-equal' else 0.)
    for p in [8, 16]: add('independent', p, rho=0., changes=0)
    add('path', 32, n=4000, changes=2); add('star', 32, n=4000)
    add('triangle-boundary', 3, rho=.6, changes=0, epsilon=0.)
    add('triangle-boundary', 3, rho=.6, changes=0, epsilon=.005)
    assert len(cells) == 38
    return cells


def population(c):
    p = c['p']; shape = c['shape']; edges = complete_edges(p)
    if shape in ('star', 'path', 'random', 'wine-topology'):
        Rs = [tree_covariance(c[t], p, c['rho']) for t in ['tree1', 'tree2']]
    elif shape in ('equicorrelated', 'near-equal', 'independent'):
        R = np.full((p, p), c['rho']); np.fill_diagonal(R, 1.)
        if shape == 'near-equal':
            R[0, 1:] += c['epsilon']; R[1:, 0] += c['epsilon']
        Rs = [R, R]
    else:
        R = np.array([[1., .6, .3+c['epsilon']], [.6, 1., .3], [.3+c['epsilon'], .3, 1.]])
        Rs = [R, R]
    taus = [2/np.pi*np.arcsin([R[a, b] for a, b in edges]) for R in Rs]
    if c['model'] == 'Binary-Gaussian': taus = [t/2 for t in taus]
    trees = [mst(p, edges, -t) for t in taus]
    if shape in ('equicorrelated', 'independent'):
        mandatory = [set(), set()]; possible = [set(range(len(edges)))]*2
        target = (0, p-1)
    elif shape == 'triangle-boundary' and c['epsilon'] == 0:
        mandatory = [{edges.index((0, 1))}]*2; possible = [set(range(len(edges)))]*2
        target = (0, 1)
    else:
        mandatory = possible = trees
        target = (len(trees[0]-trees[1]),)*2
    gaps = []; counts = []
    for t, T in zip(taus, trees):
        comparisons = fundamental_comparisons(p, edges, T)
        gaps.extend((t[comparisons[:, 0]]-t[comparisons[:, 1]]).tolist())
        counts.append(len(comparisons))
    return Rs, taus, trees, mandatory, possible, target, min(gaps), max(gaps), sum(counts)


@lru_cache(maxsize=None)
def pop_cached(cell):
    return population(revision_design()[cell])


def data_for(c, rep):
    seed = MASTER+c['cell']*100000+rep*17; rng = np.random.default_rng(seed)
    pop = pop_cached(c['cell']); xs = []
    for R in pop[0]:
        x = rng.standard_normal((c['n'], c['p'])) @ np.linalg.cholesky(R).T
        if c['model'] == 'Binary-Gaussian': x = (x > 0).astype(float)
        xs.append(x)
    return seed, xs, pop


def record(c, rep, seed, r, pop, name, seconds, **extra):
    Rs, taus, trees, mandatory, possible, target, gap_min, gap_max, true_M = pop
    covered = r['lower'] <= target[0] and target[1] <= r['upper']
    env = all(F <= mand and poss <= P for (F, P), mand, poss in zip(r['envelopes'], mandatory, possible))
    basic = {k: c[k] for k in ['cell', 'shape', 'p', 'n', 'rho', 'changes', 'model', 'epsilon']}
    return dict(**basic, rep=rep, seed=seed, method=name,
                lower=r['lower'], upper=r['upper'], width=r['upper']-r['lower'],
                detect=r['lower'] > 0, false_change=(r['lower'] > 0) if target == (0, 0) else np.nan,
                input_error=order_error(r, taus), coverage=covered, envelope_coverage=env,
                truth_lower=target[0], truth_upper=target[1], gap_min=gap_min, gap_max=gap_max,
                true_M=true_M, M=r.get('M', 0), seconds=seconds,
                optimizer_seconds=r.get('optimizer_seconds', np.nan),
                omitted=r.get('omitted', 0), fallback=r.get('fallback', False), **extra)


def expanded_one(task):
    c, rep = task; seed, xs, pop = data_for(c, rep); algorithm = seed+20261028
    start = time.perf_counter(); plans = make_plans(xs, seed=algorithm, K=5)
    plans_time = time.perf_counter()-start; rows = []
    methods = [
        ('Hoeffding', lambda: box_compare(xs), 0.),
        ('Local-McNemar', lambda: compare_legacy(xs, seed=algorithm), 0.),
        ('Local-E64', lambda: compare_legacy(xs, seed=algorithm, evidence=True, repeats=64), 0.),
        ('One-split', lambda: from_plans(plans[:1]), plans_time/5),
        ('Five-splits', lambda: from_plans(plans), plans_time),
        ('Quantile-5', lambda: from_plans(plans, quantile=True), plans_time),
        ('Pilot-E8', lambda: pilot_evidence(xs, plans[0], seed=algorithm), plans_time/5),
        ('Multiplier', lambda: multiplier_compare(xs, seed=algorithm), 0.)]
    for name, fn, overhead in methods:
        start = time.perf_counter(); result = fn(); elapsed = time.perf_counter()-start+overhead
        rows.append(record(c, rep, seed, result, pop, name, elapsed,
                           pilot_correct=np.mean([z['tree'] == T for z, T in zip(plans[0], pop[2])]),
                           pilot_M=sum(len(z['pairs']) for z in plans[0])))
    return rows


def tuning_one(task):
    c, rep = task; seed, xs, pop = data_for(c, rep); algorithm = seed+20261028; rows = []
    for gamma in [.1, .2, .3, .4, .5]:
        start = time.perf_counter(); plans = make_plans(xs, gamma=gamma, seed=algorithm, K=10)
        elapsed_plans = time.perf_counter()-start
        for K in [1, 3, 5, 10]:
            start = time.perf_counter(); r = from_plans(plans[:K]); elapsed = time.perf_counter()-start
            rows.append(record(c, rep, seed, r, pop, 'Split-union', elapsed+elapsed_plans*K/10,
                               gamma=gamma, K=K))
    return rows


def original_multiplier(task):
    ci, c, rep = task; seed = 271028041+ci*100000+rep*13; rng = np.random.default_rng(seed)
    p, n = c['p'], c['n']; edges = complete_edges(p)
    Rs = [covariance(p, c['rho'], 0), covariance(p, c['rho'], 1 if c['change'] else 0)]
    if c['model'] == 'Equicorrelated': Rs = [np.full((p, p), c['rho'])+np.eye(p)*(1-c['rho'])]*2
    xs = []; taus = []
    for R in Rs:
        x = rng.standard_normal((n, p)) @ np.linalg.cholesky(R).T
        if c['model'] == 'Student-t5': x /= np.sqrt(rng.chisquare(5, size=(n, 1))/5)
        if c['model'] == 'Binary-Gaussian': x = (x > 0).astype(float)
        xs.append(x); t = 2/np.pi*np.arcsin([R[a,b] for a,b in edges])
        taus.append(t/2 if c['model'] == 'Binary-Gaussian' else t)
    start = time.perf_counter(); r = multiplier_compare(xs, seed=seed+20261028); elapsed = time.perf_counter()-start
    trees = [mst(p, edges, -t) for t in taus]; tie = c['model'] == 'Equicorrelated'
    lo, hi = (0, p-1) if tie else (len(trees[0]-trees[1]),)*2
    env = all(not F and len(P) == len(edges) for F,P in r['envelopes']) if tie else all(F <= T <= P for (F,P),T in zip(r['envelopes'],trees))
    return [dict(cell=ci, rep=rep, seed=seed, method='Multiplier', **c, lower=r['lower'], upper=r['upper'],
                 width=r['upper']-r['lower'], detect=r['lower'] > 0, input_error=order_error(r, taus),
                 coverage=r['lower'] <= lo and hi <= r['upper'], envelope_coverage=env,
                 truth_lower=lo, truth_upper=hi, seconds=elapsed, omitted=r['omitted'],
                 optimizer_seconds=r['optimizer_seconds'], fallback=r['fallback'])]


def freeze():
    OUT.mkdir(exist_ok=True)
    cells = revision_design(); details = []
    for c in cells:
        pop = population(c)
        details.append(dict(**c, covariance1=pop[0][0].tolist(), covariance2=pop[0][1].tolist(),
                            tau1=pop[1][0].tolist(), tau2=pop[1][1].tolist(),
                            target=pop[5], gap_min=pop[6], gap_max=pop[7], true_M=pop[8]))
    config = dict(master_seed=MASTER, replications=250, alpha=.05, bootstrap_draws=499,
                  selected_matchings=8, local_matchings=64, main_gamma=.2, main_K=5,
                  tuning_cells=[5, 11, 13, 14, 23, 25],
                  tuning_gamma=[.1, .2, .3, .4, .5], tuning_K=[1, 3, 5, 10],
                  status='Revision design fixed after original results and referee report, before running new study; not preregistered',
                  original_tuning='gamma=.2 and K=5 fixed before original simulation run, after exploratory wine development; not an independent data-blind choice',
                  cells=details)
    path = OUT/'revision_protocol.json'
    text = json.dumps(config, indent=2)
    if path.exists() and path.read_text() != text: raise RuntimeError('Frozen protocol differs')
    path.write_text(text)
    print('Frozen', len(cells), 'cells; sha256', hashlib.sha256(text.encode()).hexdigest(), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['freeze','expanded','tuning','original-bootstrap','benchmark'])
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--reps', type=int, default=250)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    if a.mode == 'freeze': freeze(); return
    if not (OUT/'revision_protocol.json').exists(): raise RuntimeError('Freeze protocol first')
    cells = revision_design()
    if a.mode == 'benchmark':
        t = time.perf_counter(); result = expanded_one((cells[19], 100000))
        print([(r['method'], round(r['seconds'],3)) for r in result]); print('total', time.perf_counter()-t); return
    if a.mode == 'expanded': fn = expanded_one; tasks = [(c,r) for c in cells for r in range(a.reps)]
    elif a.mode == 'tuning':
        fn = tuning_one; selected = json.loads((OUT/'revision_protocol.json').read_text())['tuning_cells']
        tasks = [(cells[i],r) for i in selected for r in range(a.reps)]
    else: fn = original_multiplier; tasks = [(i,c,r) for i,c in enumerate(original_design()) for r in range(a.reps)]
    destination = OUT/(a.mode+'_replications.csv'); completed = set()
    if destination.exists():
        old = pd.read_csv(destination); completed = set(zip(old.cell, old.rep))
    tasks = [t for t in tasks if ((t[0], t[2]) if a.mode == 'original-bootstrap' else (t[0]['cell'],t[1])) not in completed]
    start = time.perf_counter(); buffer = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, rows in enumerate(pool.map(fn, tasks, chunksize=2)):
            buffer += rows
            if (i+1)%50 == 0 or i+1 == len(tasks):
                pd.DataFrame(buffer).to_csv(destination, mode='a', header=not destination.exists(), index=False)
                buffer = []; print(a.mode, i+1, '/',len(tasks), 'datasets',round(time.perf_counter()-start,1),'s',flush=True)
    print('Completed', a.mode, flush=True)


if __name__ == '__main__': main()
