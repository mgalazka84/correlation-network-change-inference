"""Independent direct-kernel checks of inherited baseline implementations."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import binom
from legacy_comparators import legacy_plan
from network_bounds import complete_edges

def main():
    checks=0;worst=0.
    for discrete in [False,True]:
        for n,p,seed in [(24,5,91),(51,6,218),(12,4,37)]:
            x=np.random.default_rng(seed+7).normal(size=(n,p))
            if discrete:x=np.round(x)
            for evidence in [False,True]:
                plan=legacy_plan(x,seed=seed,evidence=evidence,repeats=3)
                pairs=plan['pairs'];ends=np.asarray(complete_edges(p));N=n//2;rng=np.random.default_rng(seed);Es=[]
                fractions=[1/32,1/16,1/8,1/4,1/2,3/4,min(.5,N**-.5)]
                for r in range(3 if evidence else 1):
                    idx=rng.permutation(n)[:2*N];sign=np.sign(x[idx[::2]]-x[idx[1::2]])
                    if not evidence:sign=np.where(sign==0,np.where(rng.random(sign.shape)<.5,-1,1),sign)
                    H=sign[:,ends[:,0]]*sign[:,ends[:,1]]
                    Z=(H[:,pairs[:,0]]-H[:,pairs[:,1]])/2
                    if evidence:
                        for lam in fractions:Es.append(np.prod(1+lam*Z,axis=0))
                    else:
                        A=(Z==1).sum(axis=0);D=(Z!=0).sum(axis=0);expected=binom.sf(A-1,D,.5);expected[D==0]=1
                if evidence:expected=np.minimum(1,1/np.mean(Es,axis=0))
                err=float(np.max(np.abs(expected-plan['pvalues'])));worst=max(worst,err)
                assert np.allclose(expected,plan['pvalues'],atol=1e-13,rtol=1e-13)
                checks+=len(expected)
    (Path(__file__).resolve().parents[1]/'results').mkdir(parents=True,exist_ok=True)
    result=dict(direct_kernel_comparisons=checks,maximum_absolute_pvalue_error=worst,all_passed=True)
    (Path(__file__).resolve().parents[1]/'results'/'legacy_validation.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
