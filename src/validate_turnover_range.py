"""Exhaustive independent validation of the new upper endpoint and certificates."""
import argparse
import json
from pathlib import Path
import numpy as np
from finite_calibration import kendall_tau_a, finite_band
from calibration import kendall_scores_chunked
from network_bounds import complete_edges, interval_envelope, turnover_upper_bound, turnover_lower_bound
from validate_bounds import all_trees


def main(out):
    rng=np.random.default_rng(96291);count=0
    for p,reps in [(3,80),(4,120),(5,100)]:
        edges=complete_edges(p);trees=all_trees(p,edges)
        for _ in range(reps):
            envs=[];families=[]
            for s in range(2):
                lo=rng.integers(0,5,len(edges));hi=lo+rng.integers(0,4,len(edges))
                f,poss=interval_envelope(p,edges,lo,hi)
                envs.append((f,poss));families.append([t for t in trees if f<=t<=poss])
            expected=max(len(a-b) for a in families[0] for b in families[1])
            cert=turnover_upper_bound(p,edges,*envs,certificate=True)
            assert cert['turnover_upper']==expected==cert['rank_sum']
            # Independent verification directly from enumerated bases, not the rank implementation.
            A=set(cert['partition_A']);B=set(cert['ground'])-A
            r1=max(len(t&A) for t in families[0])
            r2dual=max(len(B-t) for t in families[1])
            assert r1+r2dual==expected
            I=set(cert['common_independent'])
            assert any(I<=t for t in families[0]) and any(not I&t for t in families[1])
            lower=turnover_lower_bound(p,edges,*envs)
            achieved={len(a-b) for a in families[0] for b in families[1]}
            assert achieved==set(range(lower,expected+1))
            count+=1
    for i in range(30):
        x=rng.integers(0,4,(20,5)) if i%2 else rng.normal(size=(25,5))
        if i%3==0:x[:,0]=1
        a=kendall_tau_a(x);b=kendall_scores_chunked(x)[0]
        assert np.max(np.abs(a-b))<1e-12
    trivial=finite_band(np.ones((8,3)),spacing=2,beta_upper=1)
    assert np.all(trivial['lower']==-1) and np.all(trivial['upper']==1)
    result=dict(status='passed',upper_optima=count,independent_rank_certificates=count,
                contiguous_ranges=count,tau_a_comparisons=30,exhausted_budget='full interval')
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='validation/turnover_range.json')
    main(parser.parse_args().out)
