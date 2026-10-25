"""Fresh-seed contrast and grouped-t follow-up, motivated by the first audit.

This is a subsequent experiment, not a preregistered part of the first study.
Its full fixed design is saved before data generation. No favorable cells are
selected for reporting and no tuning constants are fitted to its outcomes.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_pilot import covariance,gaussian_ar
from calibration import gaussian_bands,kendall_scores_chunked
from batch_calibration import batch_t_bands
from contrast_inference import contrast_bands,compare_contrasts
from network_bounds import complete_edges,mst,degrees,compare_intervals
from run_validation_study import summarize,make_cov


def main(args):
    out=Path(__file__).resolve().parents[1]/'results'/'targeted_study';out.mkdir(parents=True,exist_ok=True)
    cells=[]
    for n in [250,1000]:
        for case,kind,hub in [('clear_null','clear',0),('clear_change','clear',1),('near_null','near',0),('near_change','near',1),('exact_tie_null','tie',0)]:
            cells.append(dict(n=n,phi=.6,case=case,kind=kind,hub=hub,increment=None))
    for n,phi in [(1000,.85),(4000,.6)]:
        for hub in [0,1]:
            cells.append(dict(n=n,phi=phi,case='clear_null' if hub==0 else 'clear_change',kind='clear',hub=hub,increment=None))
    cells.append(dict(n=1000,phi=.6,case='gap_0.10_change',kind='gap',hub=1,increment=.10))
    meta=dict(seed=args.seed,reps=args.reps,draws=args.draws,p=8,alpha=.05,cells=cells,
              methods=['parzen_box','parzen_contrasts','batch_t_8'],groups=8,
              status='Follow-up design fixed after the original box audit and before these draws.')
    (out/'design.json').write_text(json.dumps(meta,indent=2))
    rows=[];start=time.time();p=8;edges=complete_edges(p);idx=np.triu_indices(p,1)
    for cell,design in enumerate(cells):
        n=design['n'];phi=design['phi']
        cs=[make_cov(p,h,design['kind'],design['increment']) for h in [0,design['hub']]]
        target=[2/np.pi*np.arcsin(c[idx]) for c in cs]
        trees=[mst(p,edges,1-t) for t in target]
        true_h=0 if design['kind']=='tie' else len(trees[0]-trees[1])
        delta=degrees(p,edges,trees[1])-degrees(p,edges,trees[0])
        for rep in range(args.reps):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,cell,rep]))
            xs=[gaussian_ar(n,c,phi,rng) for c in cs]
            fitted=[kendall_scores_chunked(x) for x in xs]
            taus=[f[0] for f in fitted];scores=[f[1] for f in fitted]
            box=gaussian_bands(scores,draws=args.draws,rng=rng)
            contrasts=contrast_bands(scores,draws=args.draws,rng=rng)
            batch=batch_t_bands(xs,groups=8)
            plugin=[mst(p,edges,1-t) for t in taus];plugin_h=len(plugin[0]-plugin[1])
            for method,bands in [('parzen_box',box),('parzen_contrasts',contrasts),('batch_t_8',batch)]:
                if method=='parzen_contrasts':
                    cover=True
                    for truth,estimate,b in zip(target,taus,bands):
                        err=truth-estimate
                        cover=cover and np.all(np.abs(err[:,None]-err[None,:])<=b['halfwidth']+1e-14)
                    graph=compare_contrasts(p,edges,taus,bands)
                else:
                    centres=[b['centre'] for b in bands] if method=='batch_t_8' else taus
                    lo=[np.maximum(-1,t-b['halfwidth']) for t,b in zip(centres,bands)]
                    hi=[np.minimum(1,t+b['halfwidth']) for t,b in zip(centres,bands)]
                    cover=all(np.all((t>=l)&(t<=u)) for t,l,u in zip(target,lo,hi))
                    graph=compare_intervals(p,edges,1-hi[0],1-lo[0],1-hi[1],1-lo[1])
                dlo,dhi=graph['degree_change_lower'],graph['degree_change_upper']
                good=(np.all(dlo<=-(p-2)) and np.all(dhi>=p-2)) if design['kind']=='tie' else np.all((dlo<=delta)&(delta<=dhi))
                valid=bool(good and graph['turnover_lower']<=true_h)
                assert not cover or valid
                rows.append(dict(cell=cell,case=design['case'],n=n,p=p,phi=phi,rep=rep,method=method,
                    true_turnover=true_h,box_covered=int(cover),graph_valid=int(valid),detected=int(graph['turnover_lower']>0),
                    turnover_lower=graph['turnover_lower'],degree_width=float(np.mean(dhi-dlo)),
                    raw_degree_width=float(np.mean(graph['raw_degree_change_upper']-graph['raw_degree_change_lower'])),
                    degree_changes_certified=int(np.sum((dlo>0)|(dhi<0))),mean_halfwidth=float(np.mean([b['halfwidth'] for b in bands])),
                    plugin_changed=int(plugin_h>0),plugin_turnover=plugin_h))
        frame=pd.DataFrame(rows);frame.to_csv(out/'replicates.csv',index=False)
        summarize(frame).to_csv(out/'summary.csv',index=False)
        print(f"cell {cell+1}/{len(cells)} n={n} phi={phi} {design['case']}: {time.time()-start:.1f}s",flush=True)
    meta['elapsed_seconds']=time.time()-start
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=200)
    parser.add_argument('--draws',type=int,default=1999)
    parser.add_argument('--seed',type=int,default=20261106)
    with threadpool_limits(limits=1):main(parser.parse_args())
