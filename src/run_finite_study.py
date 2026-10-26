"""Fixed-design finite-sample and two-sided-turnover evaluation.

Run with --out to place results outside this code-only checkout.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from finite_calibration import finite_band, gaussian_ar_spacing
from network_bounds import complete_edges, compare_intervals, turnover_upper_bound
from run_pilot import covariance, gaussian_ar, wilson


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cells=[dict(n=n,kind=kind,change=change,phi=0.) for n in [250,1000,4000,16000]
           for kind,change in [('clear',False),('clear',True),('tie',False)]]
    cells += [dict(n=1000,kind='binary',change=False,phi=0.)]
    cells += [dict(n=4000,kind='clear',change=change,phi=phi)
              for phi in [.6,.85] for change in [False,True]]
    design=dict(seed=args.seed,reps=args.reps,p=8,alpha=.05,cells=cells,
                note='Fixed before fresh draws; known AR parameter used only in dependent simulation cells.')
    (out/'design.json').write_text(json.dumps(design,indent=2))
    p=8;edges=complete_edges(p);idx=np.triu_indices(p,1);rows=[];start=time.time()
    for cell,d in enumerate(cells):
        n,phi=d['n'],d['phi'];kind=d['kind']
        cs=[covariance(p,h,kind) for h in [0,int(d['change'])]]
        target=[np.zeros(len(edges)) if kind=='binary' else 2/np.pi*np.arcsin(c[idx]) for c in cs]
        truth=compare_intervals(p,edges,1-target[0],1-target[0],1-target[1],1-target[1])
        true_min=truth['turnover_lower'];true_max=turnover_upper_bound(p,edges,truth['envelope1'],truth['envelope2'])
        spacing,beta=gaussian_ar_spacing(n,p,phi) if phi else (1,0.)
        for rep in range(args.reps):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,cell,rep]))
            xs=[rng.integers(0,2,(n,p)) if kind=='binary' else gaussian_ar(n,c,phi,rng) for c in cs]
            bands=[finite_band(x,spacing=spacing,beta_upper=beta) for x in xs]
            graph=compare_intervals(p,edges,1-bands[0]['upper'],1-bands[0]['lower'],1-bands[1]['upper'],1-bands[1]['lower'])
            upper=turnover_upper_bound(p,edges,graph['envelope1'],graph['envelope2'])
            cover=all(np.all((b['lower']<=t)&(t<=b['upper'])) for b,t in zip(bands,target))
            degree_valid=np.all(graph['degree_change_lower']<=truth['degree_change_lower']) and np.all(graph['degree_change_upper']>=truth['degree_change_upper'])
            valid=graph['turnover_lower']<=true_min and upper>=true_max and degree_valid
            assert not cover or valid
            rows.append(dict(cell=cell,n=n,phi=phi,kind=kind,change=d['change'],rep=rep,
                covered=int(cover),graph_valid=int(valid),detected=int(graph['turnover_lower']>0),
                exact_recovered=int(graph['turnover_lower']==true_min and upper==true_max),
                stability_k1=int(upper<=1),lower=graph['turnover_lower'],upper=upper,
                true_min=true_min,true_max=true_max,halfwidth=bands[0]['halfwidth'],
                retained=bands[0]['retained'],spacing=spacing,coupling_error=bands[0]['coupling_error']))
        pd.DataFrame(rows).to_csv(out/'replicates.csv',index=False)
        print(f'cell {cell+1}/{len(cells)}: {time.time()-start:.1f}s',flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for cell,group in frame.groupby('cell'):
        row={k:group.iloc[0][k] for k in ['cell','n','phi','kind','change','true_min','true_max','halfwidth','retained','spacing','coupling_error']}
        for metric in ['covered','graph_valid','detected','exact_recovered','stability_k1']:
            row[metric]=group[metric].mean();lo,hi=wilson(group[metric].sum(),len(group))
            row[metric+'_lo']=lo;row[metric+'_hi']=hi
        row['mean_lower']=group.lower.mean();row['mean_upper']=group.upper.mean();summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    design['elapsed_seconds']=time.time()-start
    (out/'metadata.json').write_text(json.dumps(design,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results/finite_study')
    parser.add_argument('--reps',type=int,default=300);parser.add_argument('--seed',type=int,default=20261026)
    with threadpool_limits(limits=1):main(parser.parse_args())
