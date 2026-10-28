"""Frozen factorial comparison; no simulation cells are selected by their results."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import norm
from paired_order import compare,compare_all,compare_multi
from finite_calibration import finite_band
from network_bounds import complete_edges,interval_envelope,turnover_lower_bound,turnover_upper_bound,mst


def covariance(p,rho,hub):
    R=np.full((p,p),rho*rho);R[hub,:]=rho;R[:,hub]=rho;np.fill_diagonal(R,1.)
    return R


def box_compare(xs):
    p=xs[0].shape[1];edges=complete_edges(p);bands=[finite_band(x) for x in xs]
    envs=[interval_envelope(p,edges,-b['upper'],-b['lower']) for b in bands]
    return dict(lower=turnover_lower_bound(p,edges,*envs),upper=turnover_upper_bound(p,edges,*envs),
                envelopes=envs,plans=[],M=2*len(edges),rejected=0,bands=bands)


def design():
    cells=[]
    for p,n,rho,model,change in itertools.product([8,16],[400,1000],[.35,.65],['Gaussian','Student-t5'],[False,True]):
        cells.append(dict(p=p,n=n,rho=rho,model=model,change=change))
    for p,change in itertools.product([8,16],[False,True]):
        cells.append(dict(p=p,n=1000,rho=.65,model='Binary-Gaussian',change=change))
    for p in [8,16]:cells.append(dict(p=p,n=1000,rho=.35,model='Equicorrelated',change=False))
    return cells


def order_error(result,taus):
    if 'parts' in result:return any(order_error(z,taus) for z in result['parts'])
    for s,plan in enumerate(result.get('plans',[])):
        pp=plan['pairs'][plan['rejected']]
        if len(pp) and np.any(taus[s][pp[:,0]]<=taus[s][pp[:,1]]+1e-12):return True
    if 'bands' in result:
        return any(np.any((tau<b['lower']-1e-12)|(tau>b['upper']+1e-12)) for tau,b in zip(taus,result['bands']))
    return False


def one(task):
    cell_id,c,rep,master=task
    seed=master+cell_id*100000+rep*13
    rng=np.random.default_rng(seed);p,n=c['p'],c['n'];rho=c['rho'];edges=complete_edges(p)
    Rs=[covariance(p,rho,0),covariance(p,rho,1 if c['change'] else 0)]
    if c['model']=='Equicorrelated':
        Rs=[np.full((p,p),rho)+np.eye(p)*(1-rho) for _ in range(2)]
    xs=[];taus=[]
    for R in Rs:
        x=rng.standard_normal((n,p))@np.linalg.cholesky(R).T
        if c['model']=='Student-t5':x/=np.sqrt(rng.chisquare(5,size=(n,1))/5)
        if c['model']=='Binary-Gaussian':x=(x>0).astype(float)
        xs.append(x)
        tau=2/np.pi*np.arcsin(np.array([R[a,b] for a,b in edges]))
        if c['model']=='Binary-Gaussian':tau/=2
        taus.append(tau)
    trees=[mst(p,edges,-t) for t in taus]
    exact=len(trees[0]-trees[1]);tie=c['model']=='Equicorrelated'
    out=[]
    for name,fn in [('Hoeffding',box_compare),('All-pairs',compare_all),('One-split',compare),('Five-splits',compare_multi)]:
        start=time.perf_counter()
        r=fn(xs) if name=='Hoeffding' else fn(xs,seed=seed+20261028)
        elapsed=time.perf_counter()-start
        if tie:
            covered=r['lower']==0 and r['upper']==p-1
            env_covered=all(not F and len(P)==len(edges) for F,P in r['envelopes'])
        else:
            covered=r['lower']<=exact<=r['upper']
            env_covered=all(F<=T<=P for (F,P),T in zip(r['envelopes'],trees))
        out.append(dict(cell=cell_id,rep=rep,seed=seed,method=name,**c,
            truth_lower=0 if tie else exact,truth_upper=p-1 if tie else exact,
            lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower'],
            input_error=order_error(r,taus),envelope_coverage=env_covered,
            coverage=covered,detect=r['lower']>0,seconds=elapsed,M=r['M'],
            forced1=len(r['envelopes'][0][0]),possible1=len(r['envelopes'][0][1]),
            forced2=len(r['envelopes'][1][0]),possible2=len(r['envelopes'][1][1]),
            fallback=r.get('fallback',False)))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));ap.add_argument('--reps',type=int,default=250);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    root=Path(a.root);(root/'results').mkdir(parents=True,exist_ok=True);cells=design();master=271028041
    if a.pilot:cells=cells[:1]
    config=dict(master_seed=master,replications=a.reps,alpha=.05,pilot_fraction=.2,repeated_splits=5,cells=cells,
      selection='Factorial design fixed in code before Monte Carlo execution; developed after exploratory wine analysis; not preregistered')
    stem='pilot' if a.pilot else 'simulation'
    (root/'results'/f'{stem}_config.json').write_text(json.dumps(config,indent=2))
    tasks=[(i,c,r,master) for i,c in enumerate(cells) for r in range(a.reps)];rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for j,result in enumerate(ex.map(one,tasks,chunksize=5)):
            rows+=result
            if (j+1)%250==0:print(f'{j+1}/{len(tasks)} datasets; {time.perf_counter()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'results'/f'{stem}_replications.csv',index=False)
    summary=df.groupby(['cell','p','n','rho','model','change','method'],sort=False).agg(
      reps=('rep','size'),coverage=('coverage','mean'),envelope_coverage=('envelope_coverage','mean'),
      input_error=('input_error','mean'),detect=('detect','mean'),lower=('lower','mean'),upper=('upper','mean'),
      width=('width','mean'),seconds=('seconds','mean'),M=('M','mean'),fallback=('fallback','sum')).reset_index()
    for name in ['coverage','input_error','detect']:
        rate=summary[name];N=summary.reps;z=norm.ppf(.975);den=1+z*z/N
        center=(rate+z*z/(2*N))/den;rad=z*np.sqrt(rate*(1-rate)/N+z*z/(4*N*N))/den
        summary[name+'_lo']=center-rad;summary[name+'_hi']=center+rad
    summary.to_csv(root/'results'/f'{stem}_summary.csv',index=False)
    print('Finished',len(tasks),'datasets in',time.perf_counter()-start,'seconds',flush=True)

if __name__=='__main__':main()
