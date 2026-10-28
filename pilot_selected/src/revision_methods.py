"""Referee-requested calibrations; established tests and aggregation rules.

The multiplier comparator is asymptotic. Exact methods assume IID rows and
fixed tuning. Finite quantile aggregation still depends on the random splits.
"""
import time
import numpy as np
from scipy.special import logsumexp
from paired_order import (order_plan, holm, envelope_from_rejections,
                          all_order_envelope)
from legacy_comparators import local_pairs, closure_pairs
from network_bounds import (complete_edges, forest, components,
                            turnover_lower_bound, turnover_upper_bound)


def adjusted_holm(pv):
    pv = np.asarray(pv, float)
    ix = np.argsort(pv, kind='stable')
    adj = np.ones(len(pv))
    if len(pv):
        adj[ix] = np.minimum(1., np.maximum.accumulate(
            pv[ix] * np.arange(len(pv), 0, -1)))
    return adj


def finish(p, envs, **metadata):
    edges = complete_edges(p)
    fallback = any(not F <= P or not forest(p, edges, F) or
                   components(p, edges, P) != 1 for F, P in envs)
    if fallback:
        envs = [(set(), set(range(len(edges)))) for _ in range(2)]
    start = time.perf_counter()
    lo = turnover_lower_bound(p, edges, *envs)
    hi = turnover_upper_bound(p, edges, *envs)
    return dict(lower=lo, upper=hi, envelopes=envs, fallback=fallback,
                optimizer_seconds=time.perf_counter()-start, **metadata)


def make_plans(xs, gamma=.2, seed=1, K=5):
    return [[order_plan(x, gamma, seed+1000003*k+104729*s)
             for s, x in enumerate(xs)] for k in range(K)]


def from_plans(splits, alpha=.05, quantile=False):
    """Union of envelopes, or fixed-quantile aggregation of Holm p-values.

For the quantile method, a nonselected directed hypothesis is assigned one.
The (ceil(K/2))-th smallest adjusted p-value is divided by ceil(K/2)/K.
The resulting whole family controls FWER by a Markov counting argument.
"""
    K = len(splits); p = splits[0][0]['p']; m = len(splits[0][0]['edges'])
    if quantile:
        all_adj = np.ones((K, 2, m, m), float)
    parts = []; M = 0
    for k, plans in enumerate(splits):
        adj = adjusted_holm(np.concatenate([z['pvalues'] for z in plans]))
        start = 0; envs = []; copied = []
        for s, z in enumerate(plans):
            count = len(z['pvalues']); q = adj[start:start+count]; start += count
            if quantile:
                f, e = z['pairs'].T
                all_adj[k, s, f, e] = q
            keep = q <= alpha/K
            copied.append(dict(z, rejected=keep))
            envs.append(envelope_from_rejections(z, keep))
        parts.append(dict(plans=copied, envelopes=envs)); M += len(adj)
    if quantile:
        r = (K+1)//2
        q = np.minimum(1., np.partition(all_adj, r-1, axis=0)[r-1]/(r/K))
        plans = []; envs = []
        for s in range(2):
            rel = np.argwhere(q[s] <= alpha)
            plan = dict(p=p, edges=complete_edges(p), pairs=rel,
                        rejected=np.ones(len(rel), bool))
            plans.append(plan)
            closure = closure_pairs(rel, plan['rejected'], m)
            envs.append(all_order_envelope(dict(plan, pairs=closure),
                                          np.ones(len(closure), bool)))
        return finish(p, envs, plans=plans, M=M, aggregation='fixed-quantile')
    envs = [(set().union(*(part['envelopes'][s][0] for part in parts)),
             set.intersection(*(part['envelopes'][s][1] for part in parts)))
            for s in range(2)]
    return finish(p, envs, parts=parts, M=M, aggregation='Bonferroni-union')


def pilot_evidence(xs, plans, alpha=.05, seed=1, matchings=8):
    """The companion's raw-kernel e-average on the selected validation family."""
    output = []
    for s, (x, plan) in enumerate(zip(xs, plans)):
        v = x[plan['validation_indices']]
        N = len(v)//2; ends = np.asarray(plan['edges']); f, e = plan['pairs'].T
        rng = np.random.default_rng(seed+104729*s+7717)
        lam = np.array([1/32, 1/16, 1/8, 1/4, 1/2, 3/4, min(.5, N**-.5)])
        logs = []
        for _ in range(matchings):
            ix = rng.permutation(len(v))[:2*N]
            sign = np.sign(v[ix[::2]]-v[ix[1::2]])
            H = sign[:, ends[:, 0]]*sign[:, ends[:, 1]]
            Z = (H[:, f]-H[:, e])/2
            counts = np.stack([(Z == z).sum(axis=0) for z in [1, -1, .5, -.5]], axis=1)
            logs.append(counts @ np.log1p(np.array([1, -1, .5, -.5])[:, None]*lam))
        logE = logsumexp(np.concatenate(logs, axis=1), axis=1)-np.log(7*matchings)
        output.append(dict(plan, pvalues=np.exp(np.minimum(0., -logE))))
    keep = holm(np.concatenate([z['pvalues'] for z in output]), alpha)
    envs = []; start = 0
    for z in output:
        z['rejected'] = keep[start:start+len(z['pvalues'])]; start += len(z['pvalues'])
        envs.append(envelope_from_rejections(z, z['rejected']))
    return finish(xs[0].shape[1], envs, plans=output, M=start)


def kernel_projection(x, block=32):
    """Exact rowwise tau-a kernel means, including ties; bounded-memory BLAS."""
    n, p = x.shape; a, b = np.triu_indices(p, 1)
    g = np.empty((n, len(a)))
    for start in range(0, n, block):
        signs = np.sign(x[start:start+block, None, :]-x[None, :, :])
        cross = signs.transpose(0, 2, 1) @ signs
        g[start:start+block] = cross[:, a, b]/(n-1)
    tau = g.mean(axis=0)
    return tau, 2*(g-tau)


def multiplier_compare(xs, alpha=.05, seed=1, B=499):
    """Full-U Gaussian jackknife multiplier on fixed adjacent-edge contrasts.

One pooled maximum controls both groups asymptotically. Zero estimated
variance comparisons are omitted. This is not an exact finite-sample test.
"""
    p = xs[0].shape[1]; edges = complete_edges(p); pairs = local_pairs(edges)
    unique = pairs[pairs[:, 0] < pairs[:, 1]]; f, e = unique.T
    rng = np.random.default_rng(seed+141421); maxima = np.zeros(B); plans = []
    for x in xs:
        n = len(x); tau, psi = kernel_projection(x)
        cov = psi.T @ psi / n**2
        var = np.maximum(0., cov[f, f]+cov[e, e]-2*cov[f, e])
        se = np.sqrt(var); valid = se > 1e-12
        G = (rng.standard_normal((B, n)) @ psi)/n
        for start in range(0, len(f), 1024):
            sl = np.arange(start, min(len(f), start+1024)); sl = sl[valid[sl]]
            if len(sl):
                maxima = np.maximum(maxima, np.max(np.abs(G[:, f[sl]]-G[:, e[sl]])/se[sl], axis=1))
        plans.append(dict(p=p, edges=edges, pairs=pairs, tau=tau, se=se,
                          valid=valid, omitted=int((~valid).sum())))
    # Conservative empirical order-statistic quantile; no finite-B exact claim.
    rank = min(B-1, int(np.ceil((B+1)*(1-alpha)))-1)
    critical = np.sort(maxima)[rank]; envs = []
    for z in plans:
        diff = z['tau'][f]-z['tau'][e]; keep1 = z['valid'] & (diff > critical*z['se'])
        keep2 = z['valid'] & (-diff > critical*z['se'])
        relation = np.vstack([unique[keep1], unique[keep2][:, ::-1]])
        # Save the original strict rejections for population FWER assessment.
        z['pairs'] = relation; z['rejected'] = np.ones(len(relation), bool)
        closure = closure_pairs(relation, z['rejected'], len(edges))
        envs.append(all_order_envelope(dict(z, pairs=closure), np.ones(len(closure), bool)))
    return finish(p, envs, plans=plans, M=2*len(pairs), critical=critical,
                  omitted=sum(z['omitted'] for z in plans))
