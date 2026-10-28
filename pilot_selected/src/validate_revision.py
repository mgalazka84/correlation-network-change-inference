"""Independent arithmetic and equivalence checks for the revision code."""
from pathlib import Path
import itertools, json
import numpy as np
from paired_order import compare, compare_multi, holm, fundamental_comparisons
from finite_calibration import kendall_tau_a
from revision_methods import kernel_projection, adjusted_holm, make_plans, from_plans, multiplier_compare
from run_revision_study import revision_design, population
from network_bounds import complete_edges, forest


def main():
    rng = np.random.default_rng(812773); largest = 0.; projection_checks = 0
    for n, p, discrete in itertools.product([7, 20, 41], [3, 5, 8], [False, True]):
        x = rng.integers(0, 3, (n, p)).astype(float) if discrete else rng.normal(size=(n, p))
        tau, psi = kernel_projection(x, block=5); edges = complete_edges(p)
        brute = np.zeros((n, len(edges)))
        for i in range(n):
            for j in range(n):
                sign = np.sign(x[i]-x[j])
                brute[i] += np.array([sign[a]*sign[b] for a,b in edges])/(n-1)
        errors = [np.max(abs(tau-kendall_tau_a(x))), np.max(abs(tau-brute.mean(axis=0))),
                  np.max(abs(psi-2*(brute-brute.mean(axis=0))))]
        largest = max(largest, *errors); projection_checks += 1
        assert max(errors) < 1e-12
    holm_checks = 0
    for m in [0, 1, 3, 50, 200]:
        for _ in range(10):
            pv = rng.uniform(size=m)**4
            for a in [.001, .01, .05, .5, .999]:
                assert np.array_equal(adjusted_holm(pv) <= a, holm(pv, a)); holm_checks += 1
    split_checks = 0
    for p, seed in itertools.product([3, 5, 8], range(3)):
        xs = [rng.normal(size=(100,p)), rng.integers(0, 3, (100,p)).astype(float)]
        plans = make_plans(xs, seed=seed, K=5)
        for K in [1,5]:
            new = from_plans(plans[:K]); old = compare(xs, seed=seed) if K == 1 else compare_multi(xs, seed=seed)
            assert new['envelopes'] == old['envelopes']
            assert (new['lower'],new['upper']) == (old['lower'],old['upper']); split_checks += 1
    population_checks = 0
    for c in revision_design():
        pop = population(c)
        for R in pop[0]: assert np.linalg.eigvalsh(R).min() > 0
        if c['tree1']:
            edges = complete_edges(c['p'])
            assert [{edges[i] for i in T} for T in pop[2]] == [set(map(tuple,c[t])) for t in ['tree1','tree2']]
        if c['shape'] == 'triangle-boundary':
            edges=complete_edges(3); vals=[]; fam=[]
            for T in itertools.combinations(range(3),2):
                vals.append(sum(pop[1][0][list(T)])); fam.append(set(T))
            opts=[T for T,v in zip(fam,vals) if abs(v-max(vals))<1e-12]
            hs=[len(A-B) for A in opts for B in opts]
            assert (min(hs),max(hs)) == pop[5]
        population_checks += 1
    # The quantile-counting implication holds for arbitrary dependence.
    quantile_checks = 0
    for K in [1,3,5,10]:
        r=(K+1)//2
        for _ in range(100):
            q=rng.uniform(size=(K,17)); a=.2
            reject=np.any(np.sort(q,axis=0)[r-1] <= a*r/K)
            count=np.sum(np.min(q,axis=1) <= a*r/K)
            assert not reject or count >= r; quantile_checks += 1
    out=dict(projection_datasets=projection_checks,max_projection_error=largest,
             holm_equivalences=holm_checks,split_equivalences=split_checks,
             population_checks=population_checks,quantile_counting_checks=quantile_checks,status='passed')
    root=Path(__file__).resolve().parents[1];(root/'revision_results').mkdir(exist_ok=True)
    (root/'revision_results/revision_validation.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__ == '__main__':main()
