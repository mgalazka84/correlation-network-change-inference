"""Independent finite enumerations for calibration and graph transfer."""
import itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import binom
from paired_order import fundamental_comparisons,envelope_from_rejections,all_order_envelope,order_plan
from network_bounds import complete_edges,forest,turnover_lower_bound,turnover_upper_bound


def main():
    root=Path(__file__).resolve().parents[1];(root/'results').mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(28102026)
    # The randomized encoding preserves every possible tied kernel contrast.
    encoding=0
    for a,b in itertools.product([-1,0,1],repeat=2):
        ey=np.mean([int(u<(a+1)/2)-int(u<(b+1)/2) for u in [.25,.75]])
        assert ey==(a-b)/2;encoding+=1
    # Exact conditional null size; discreteness is retained, never mid-p.
    size_checks=0;max_excess=0.
    for D in range(101):
        k=np.arange(D+1);pv=binom.sf(k-1,D,.5)
        for pi in [0,.1,.3,.49,.5]:
            probs=binom.pmf(k,D,pi)
            for alpha in [.00001,.001,.01,.05,.1,.5]:
                size=probs[pv<=alpha].sum();max_excess=max(max_excess,float(size-alpha))
                assert size<=alpha+1e-12;size_checks+=1
    p=4;edges=complete_edges(p);m=len(edges)
    trees=[set(t) for t in itertools.combinations(range(m),p-1) if forest(p,edges,t)]
    envelopes=[];screens=0
    # All 729 discrete weight vectors include ties. All 16 pilot trees are tried.
    for w in itertools.product(range(3),repeat=m):
        vals=np.array([sum(w[e] for e in t) for t in trees]);opt=[t for t,v in zip(trees,vals) if v==min(vals)]
        for pilot in trees:
            pairs=fundamental_comparisons(p,edges,pilot)
            valid=np.array([w[f]<w[e] for f,e in pairs])
            # Randomly omit valid relations: a certificate need not be complete.
            rej=valid & (rng.random(len(pairs))<.7)
            plan=dict(p=p,edges=edges,tree=pilot,pairs=pairs)
            F,P=envelope_from_rejections(plan,rej)
            assert all(F<=t<=P for t in opt)
            assert F<=pilot<=P
            screens+=1
            if len(envelopes)<500:envelopes.append((F,P))
        pairs=np.array([(f,e) for f in range(m) for e in range(m) if f!=e])
        rej=np.array([w[f]<w[e] for f,e in pairs])&(rng.random(len(pairs))<.7)
        F,P=all_order_envelope(dict(p=p,edges=edges,pairs=pairs),rej)
        assert all(F<=t<=P for t in opt);screens+=1
    ranges=0
    for j in range(500):
        a=envelopes[int(rng.integers(len(envelopes)))];b=envelopes[int(rng.integers(len(envelopes)))]
        aa=[t for t in trees if a[0]<=t<=a[1]];bb=[t for t in trees if b[0]<=t<=b[1]]
        actual={len(x-y) for x in aa for y in bb}
        lo=turnover_lower_bound(p,edges,a,b);hi=turnover_upper_bound(p,edges,a,b)
        assert actual==set(range(lo,hi+1));ranges+=1
    x=rng.normal(size=(101,5));plan=order_plan(x,seed=99)
    assert not set(plan['pilot_indices'])&set(plan['validation_indices'])
    assert len(set(plan['validation_indices']))==2*plan['N']
    result=dict(kernel_encoding_identities=encoding,conditional_null_size_checks=size_checks,
      maximum_size_excess=max_excess,exhaustive_graph_screens=screens,enumerated_turnover_ranges=ranges,
      disjoint_split_and_pair_check=True,all_passed=True)
    (root/'results'/'validation.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
