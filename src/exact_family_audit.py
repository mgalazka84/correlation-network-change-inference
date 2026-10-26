"""Enumerate all p^(p-2) trees and audit box-family turnover relaxation.

Endpoint search is exact: it stops early only when both certified outer
endpoints have attained witnesses in the enumerated exact families.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from network_bounds import complete_edges,compare_intervals,turnover_upper_bound
from finite_calibration import finite_band
from calibration import kendall_scores_chunked,gaussian_bands
from run_pilot import covariance,gaussian_ar


def enumerate_paths(p):
    n=p**(p-2); ids=np.arange(n); seq=np.array([(ids//p**k)%p for k in range(p-2)]).T
    deg=np.ones((n,p),dtype=np.int8);rows=np.arange(n)
    for k in range(p-2):deg[rows,seq[:,k]]+=1
    edges=complete_edges(p);lookup=np.zeros((p,p),dtype=int)
    for e,(a,b) in enumerate(edges):lookup[a,b]=lookup[b,a]=e
    te=np.empty((n,p-1),dtype=np.int16)
    paths=np.zeros((n,p,p),dtype=np.uint32)
    for k in range(p-2):
        a=np.argmax(deg==1,axis=1);b=seq[:,k];e=lookup[a,b];te[:,k]=e
        paths[rows,a,b]=paths[rows,b,a]=np.left_shift(np.uint32(1),e.astype(np.uint32))
        deg[rows,a]=0;deg[rows,b]-=1
    a=np.argmax(deg==1,axis=1);deg[rows,a]=0;b=np.argmax(deg==1,axis=1);e=lookup[a,b];te[:,-1]=e
    paths[rows,a,b]=paths[rows,b,a]=np.left_shift(np.uint32(1),e.astype(np.uint32))
    for k in range(p):
        left=paths[:,:,k,None];right=paths[:,None,k,:]
        update=(paths==0)&(left!=0)&(right!=0)
        paths=np.where(update,left|right,paths)
        paths[:,np.arange(p),np.arange(p)]=0
    paths=paths[:,np.triu_indices(p,1)[0],np.triu_indices(p,1)[1]]
    masks=np.bitwise_or.reduce(np.left_shift(np.uint32(1),te.astype(np.uint32)),axis=1)
    assert len(np.unique(masks))==n
    return te,paths,masks


def possible_masks(te,paths,masks,lo,hi):
    maximum=np.full(paths.shape,-np.inf)
    for k in range(te.shape[1]):
        e=te[:,k];bit=np.left_shift(np.uint32(1),e.astype(np.uint32))
        maximum=np.maximum(maximum,np.where((paths&bit[:,None])!=0,lo[e,None],-np.inf))
    # Tree edges satisfy equality against their own upper endpoints automatically.
    return masks[np.all(maximum<=hi[None,:]+1e-12,axis=1)]


def exact_range(a,b,p,outer):
    r=p-1;lower=r;upper=0;checked=0
    if np.intersect1d(a,b).size: lower=0
    for x in a:
        counts=np.bitwise_count(np.bitwise_and(x,b))
        lower=min(lower,int(r-counts.max()));upper=max(upper,int(r-counts.min()));checked+=len(b)
        if (lower,upper)==outer:break
    assert lower>=outer[0] and upper<=outer[1]
    return lower,upper,checked


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cells=[dict(method='finite',kind='clear',n=n,change=c) for n in [1000,16000] for c in [False,True]]
    cells += [dict(method='parzen',kind=k,n=1000,change=c) for k,c in [('clear',False),('clear',True),('near',True),('tie',False)]]
    (out/'design.json').write_text(json.dumps(dict(seed=args.seed,reps=args.reps,p=8,cells=cells,trees_per_period=8**6),indent=2))
    start=time.perf_counter();te,paths,masks=enumerate_paths(8);edges=complete_edges(8);rows=[]
    for cell,d in enumerate(cells):
        for rep in range(args.reps):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,cell,rep]))
            cs=[covariance(8,h,d['kind']) for h in [0,int(d['change'])]]
            xs=[gaussian_ar(d['n'],c,0.,rng) for c in cs]
            if d['method']=='finite':bands=[finite_band(x) for x in xs];lo=[1-b['upper'] for b in bands];hi=[1-b['lower'] for b in bands]
            else:
                stats=[kendall_scores_chunked(x) for x in xs]
                bands=gaussian_bands([v[1] for v in stats],draws=1999,rng=rng)
                lo=[np.maximum(0,1-v[0]-b['halfwidth']) for v,b in zip(stats,bands)]
                hi=[np.minimum(2,1-v[0]+b['halfwidth']) for v,b in zip(stats,bands)]
            g=compare_intervals(8,edges,lo[0],hi[0],lo[1],hi[1]);upper=turnover_upper_bound(8,edges,g['envelope1'],g['envelope2'])
            exact=[possible_masks(te,paths,masks,a,b) for a,b in zip(lo,hi)]
            el,eu,checks=exact_range(*exact,8,(g['turnover_lower'],upper))
            rows.append(dict(cell=cell,rep=rep,**d,family1=len(exact[0]),family2=len(exact[1]),
                outer_lower=g['turnover_lower'],outer_upper=upper,exact_lower=el,exact_upper=eu,
                lower_gap=el-g['turnover_lower'],upper_gap=upper-eu,pairs_checked=checks))
        pd.DataFrame(rows).to_csv(out/'replicates.csv',index=False)
        print(f'exact cell {cell+1}/{len(cells)} in {time.perf_counter()-start:.1f}s',flush=True)
    f=pd.DataFrame(rows);summary=f.groupby('cell').agg(n=('rep','count'),lower_gap_count=('lower_gap',lambda x:int((x>0).sum())),
        upper_gap_count=('upper_gap',lambda x:int((x>0).sum())),max_lower_gap=('lower_gap','max'),max_upper_gap=('upper_gap','max'),
        mean_family1=('family1','mean'),mean_family2=('family2','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=25)
    p.add_argument('--seed',type=int,default=2026102602)
    with threadpool_limits(1):main(p.parse_args())
