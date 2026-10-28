"""Focused E64 sensitivity benchmark in all six strong changed-star cells.

This extra benchmark was specified after the six-method simulation results;
all six cells and all 250 paired replications are retained.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,time
import numpy as np
import pandas as pd
from run_study import design,covariance,order_error
from network_bounds import complete_edges
from legacy_comparators import compare_legacy

def one(task):
    ci,c,rep=task;seed=271028041+ci*100000+rep*13;rng=np.random.default_rng(seed)
    p,n,rho=c['p'],c['n'],c['rho'];edges=complete_edges(p);xs=[];taus=[]
    for h in [0,1]:
        R=covariance(p,rho,h);x=rng.standard_normal((n,p))@np.linalg.cholesky(R).T
        if c['model']=='Student-t5':x/=np.sqrt(rng.chisquare(5,size=(n,1))/5)
        if c['model']=='Binary-Gaussian':x=(x>0).astype(float)
        xs.append(x);t=2/np.pi*np.arcsin([R[a,b] for a,b in edges])
        taus.append(t/2 if c['model']=='Binary-Gaussian' else t)
    start=time.perf_counter();r=compare_legacy(xs,seed=seed+20261028,evidence=True,repeats=64)
    return dict(cell=ci,rep=rep,seed=seed,method='Local-E64',**c,lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower'],detect=r['lower']>0,coverage=r['lower']<=p-2<=r['upper'],input_error=order_error(r,taus),seconds=time.perf_counter()-start)

def main():
    out=Path(__file__).resolve().parents[1]/'results';out.mkdir(parents=True,exist_ok=True);cells=[(i,c) for i,c in enumerate(design()) if c['change'] and c['n']==1000 and c['rho']==.65]
    assert len(cells)==6
    (out/'e64_config.json').write_text(json.dumps(dict(cells=cells,reps=250,matchings=64,timing='Added after inspection of main results; all six strong changed-star cells retained'),indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(one,[(ci,c,r) for ci,c in cells for r in range(250)],chunksize=5)):
            rows.append(row)
            if (i+1)%250==0:
                pd.DataFrame(rows).to_csv(out/'e64_replications.csv',index=False);print(i+1,'/1500',flush=True)
    print(pd.DataFrame(rows).groupby(['p','model']).agg(detection=('detect','mean'),width=('width','mean'),coverage=('coverage','mean')).to_string())
if __name__=='__main__':main()
