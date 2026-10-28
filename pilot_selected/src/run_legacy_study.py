"""Additional inherited-method benchmarks on every cell of the fixed design."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from legacy_comparators import compare_legacy
from run_study import design,covariance,order_error
from network_bounds import complete_edges,mst

def one(task):
    ci,c,rep=task;seed=271028041+ci*100000+rep*13;rng=np.random.default_rng(seed)
    p,n,rho=c['p'],c['n'],c['rho'];edges=complete_edges(p)
    Rs=[covariance(p,rho,0),covariance(p,rho,1 if c['change'] else 0)]
    tie=c['model']=='Equicorrelated'
    if tie:Rs=[np.full((p,p),rho)+np.eye(p)*(1-rho) for _ in range(2)]
    xs=[];taus=[]
    for R in Rs:
        x=rng.standard_normal((n,p))@np.linalg.cholesky(R).T
        if c['model']=='Student-t5':x/=np.sqrt(rng.chisquare(5,size=(n,1))/5)
        if c['model']=='Binary-Gaussian':x=(x>0).astype(float)
        xs.append(x);tau=2/np.pi*np.arcsin(np.array([R[a,b] for a,b in edges]))
        if c['model']=='Binary-Gaussian':tau/=2
        taus.append(tau)
    trees=[mst(p,edges,-t) for t in taus];truth=len(trees[0]-trees[1]);rows=[]
    for label,evidence in [('Local-McNemar',False),('Local-E8',True)]:
        start=time.perf_counter();r=compare_legacy(xs,seed=seed+20261028,evidence=evidence)
        coverage=(r['lower']==0 and r['upper']==p-1) if tie else r['lower']<=truth<=r['upper']
        envcov=all(not F and len(P)==len(edges) for F,P in r['envelopes']) if tie else all(F<=T<=P for (F,P),T in zip(r['envelopes'],trees))
        rows.append(dict(cell=ci,rep=rep,seed=seed,method=label,**c,lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower'],coverage=coverage,envelope_coverage=envcov,input_error=order_error(r,taus),detect=r['lower']>0,seconds=time.perf_counter()-start,M=r['M'],fallback=r['fallback']))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=250);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    out=Path(__file__).resolve().parents[1]/'results';out.mkdir(parents=True,exist_ok=True);cells=design()
    (out/'legacy_config.json').write_text(json.dumps(dict(reps=args.reps,cells=cells,master_seed=271028041,reason='Inherited exact comparison methods added after checking the companion manuscript and before inspecting main simulation results'),indent=2))
    tasks=[(i,c,r) for i,c in enumerate(cells) for r in range(args.reps)];rows=[];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(one,tasks,chunksize=5)):
            rows.extend(result)
            if (i+1)%250==0:
                pd.DataFrame(rows).to_csv(out/'legacy_replications.csv',index=False)
                print(f'{i+1}/{len(tasks)} datasets {time.perf_counter()-t:.1f}s',flush=True)
    print('Finished',flush=True)
if __name__=='__main__':main()
