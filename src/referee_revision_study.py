"""Fixed-design dimensional, heavy-tail and volatility-clustering audit.

Outputs are written outside the source checkout. The SV finite procedure uses
an oracle latent-AR dependence bound, never an estimated market-data bound.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scipy.stats import t
from finite_calibration import finite_band, gaussian_ar_spacing
from batch_calibration import batch_t_bands
from network_bounds import (complete_edges, interval_envelope, turnover_lower_bound,
                            turnover_upper_bound, mst)
from run_pilot import wilson


def minimax(p, edges, weights):
    """Floyd--Warshall in the (min,max) semiring, for nonnegative weights."""
    d=np.full((p,p),np.inf); np.fill_diagonal(d,0.)
    for (a,b),w in zip(edges,weights): d[a,b]=d[b,a]=w
    for k in range(p): d=np.minimum(d,np.maximum(d[:,k,None],d[None,k,:]))
    return d[np.triu_indices(p,1)]


def hierarchy_interval(p, edges, lowers, uppers):
    lo=[minimax(p,edges,w) for w in lowers]
    hi=[minimax(p,edges,w) for w in uppers]
    lower=float(np.maximum(0,np.maximum(lo[0]-hi[1],lo[1]-hi[0])).max())
    upper=float(np.maximum(hi[0]-lo[1],hi[1]-lo[0]).max())
    return lower,upper


def paired_correlation(p, strength, changed):
    """Market factor variance .2; pair factor strength-.2; residual 1-strength."""
    r=np.full((p,p),.2); np.fill_diagonal(r,1.)
    for a in range(0,p-1,4):
        pairs=[(a,a+2),(a+1,a+3)] if changed else [(a,a+1),(a+2,a+3)]
        for i,j in pairs:r[i,j]=r[j,i]=strength
    assert np.linalg.eigvalsh(r).min()>0
    return r


def sample(n, r, sv, rng):
    z=rng.normal(size=(n,len(r)))@np.linalg.cholesky(r).T
    x=z/np.sqrt(rng.chisquare(5,n)/5)[:,None]
    if sv:
        v=np.empty(n);v[0]=rng.normal(scale=.4/np.sqrt(1-.95**2))
        for i in range(1,n):v[i]=.95*v[i-1]+.4*rng.normal()
        x*=np.exp(v/2)[:,None]
    return x


@lru_cache(maxsize=1024)
def graph_range(p,f1,p1,f2,p2):
    edges=complete_edges(p);envs=[(set(f1),set(p1)),(set(f2),set(p2))]
    return (turnover_lower_bound(p,edges,*envs),turnover_upper_bound(p,edges,*envs))


def range_from_boxes(p,edges,lo,hi):
    envs=[interval_envelope(p,edges,a,b) for a,b in zip(lo,hi)]
    return graph_range(p,*(frozenset(s) for env in envs for s in env))


def run_cell(task):
    cell,d,reps,seed,out=task
    with threadpool_limits(limits=1):
        p=49;n=1006;edges=complete_edges(p);idx=np.triu_indices(p,1)
        rs=[paired_correlation(p,d['strength'],False),paired_correlation(p,d['strength'],d['change'])]
        target=[2/np.pi*np.arcsin(r[idx]) for r in rs]
        true_h=range_from_boxes(p,edges,[1-x for x in target],[1-x for x in target])
        true_u=float(np.abs(minimax(p,edges,1-target[0])-minimax(p,edges,1-target[1])).max())
        spacing,beta=gaussian_ar_spacing(n,1,.95) if d['sv'] else (1,0.)
        rows=[];start=time.perf_counter()
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([seed,cell,rep]))
            xs=[sample(n,r,d['sv'],rng) for r in rs]
            bands=[finite_band(x,spacing=spacing,beta_upper=beta) for x in xs]
            lo=[1-b['upper'] for b in bands];hi=[1-b['lower'] for b in bands]
            lh,uh=range_from_boxes(p,edges,lo,hi)
            ul,uu=hierarchy_interval(p,edges,lo,hi)
            cover=all(np.all((b['lower']<=a)&(a<=b['upper'])) for b,a in zip(bands,target))
            valid=lh<=true_h[0] and uh>=true_h[1] and ul<=true_u+1e-12 and uu>=true_u-1e-12
            assert not cover or valid
            batch=batch_t_bands(xs)
            batch_cover=all(np.all(np.abs(b['centre']-a)<=b['halfwidth']) for b,a in zip(batch,target))
            rows.append(dict(cell=cell,rep=rep,**d,n=n,p=p,covered=int(cover),graph_valid=int(valid),
                detected=int(lh>0),lower=lh,upper=uh,true_min=true_h[0],true_max=true_h[1],
                hierarchy_detected=int(ul>0),hierarchy_lower=ul,hierarchy_upper=uu,
                hierarchy_true=true_u,batch_covered=int(batch_cover),
                halfwidth=bands[0]['halfwidth'],spacing=spacing,retained=bands[0]['retained'],
                coupling_error=bands[0]['coupling_error']))
        frame=pd.DataFrame(rows);frame.to_csv(Path(out)/f'cell_{cell}.csv',index=False)
        print(f'cell {cell}: {reps} repetitions in {time.perf_counter()-start:.1f}s',flush=True)
        return rows


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cells=[dict(strength=q,sv=sv,change=change) for q,sv in [(.7,False),(.94,False),(.94,True)] for change in [False,True]]
    design=dict(seed=args.seed,reps=args.reps,cells=cells,n=1006,p=49,df=5,
        market_variance=.2,sv_phi=.95,sv_innovation_sd=.4,alpha=.05,
        note='Fixed before evaluation. Strong-pair benchmark is specialized, not calibrated to industry returns.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    tasks=[(i,d,args.reps,args.seed,str(out)) for i,d in enumerate(cells)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=sum(list(pool.map(run_cell,tasks)),[])
    frame=pd.DataFrame(rows);frame.to_csv(out/'replicates.csv',index=False)
    summary=[]
    for cell,g in frame.groupby('cell'):
        r={k:g.iloc[0][k] for k in ['cell','strength','sv','change','n','p','true_min','true_max','halfwidth','spacing','retained','coupling_error']}
        for k in ['covered','graph_valid','detected','hierarchy_detected','batch_covered']:
            r[k]=g[k].mean();r[k+'_lo'],r[k+'_hi']=wilson(g[k].sum(),len(g))
        for k in ['lower','upper','hierarchy_lower','hierarchy_upper']:r['mean_'+k]=g[k].mean()
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--reps',type=int,default=100);parser.add_argument('--seed',type=int,default=2026102601)
    parser.add_argument('--workers',type=int,default=3)
    main(parser.parse_args())
