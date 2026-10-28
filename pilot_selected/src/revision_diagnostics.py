"""Fixed auxiliary checks: wine seed distribution, pairing variance and timings."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time,zipfile
import numpy as np
import pandas as pd
from paired_order import compare,compare_multi
from revision_methods import make_plans,from_plans,pilot_evidence,multiplier_compare,kernel_projection
from legacy_comparators import compare_legacy
from run_study import box_compare
from run_revision_study import revision_design,data_for

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_results'


def wine():
    with zipfile.ZipFile(ROOT/'data/wine_quality.zip') as z:
        dfs=[pd.read_csv(z.open(f'winequality-{colour}.csv'),sep=';').drop(columns='quality') for colour in ['red','white']]
    rows=[]
    old=pd.read_csv(ROOT/'results/wine_seed_sensitivity.csv')
    for policy in ['All records','Unique feature rows']:
        xs=[(x.drop_duplicates() if policy.startswith('Unique') else x).to_numpy() for x in dfs]
        for seed in range(20261028,20261048):
            plans=make_plans(xs,seed=seed,K=5)
            for name,fn in [('One-split',lambda:from_plans(plans[:1])),
                            ('Five-splits',lambda:from_plans(plans)),
                            ('Quantile-5',lambda:from_plans(plans,quantile=True)),
                            ('Pilot-E8',lambda:pilot_evidence(xs,plans[0],seed=seed))]:
                r=fn();rows.append(dict(policy=policy,method=name,seed=seed,lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower']))
                if name in ['One-split','Five-splits']:
                    z=old[(old.policy==policy)&(old.method==name)&(old.seed==seed)].iloc[0]
                    assert (r['lower'],r['upper'])==(z.lower,z.upper)
    df=pd.DataFrame(rows);df.to_csv(OUT/'wine_seed_distribution.csv',index=False)
    df.groupby(['policy','method','lower','upper']).size().rename('count').reset_index().to_csv(OUT/'wine_interval_frequencies.csv',index=False)
    print('Wine checks passed; all 80 original outputs reproduced; 80 additional outputs',flush=True)


def variance():
    n=400;replications=2000;rng=np.random.default_rng(281028903);rows=[]
    for rep in range(replications):
        x=rng.standard_normal((n,3));tau,_=kernel_projection(x)
        # Complete-edge order (0,1),(0,2),(1,2); fixed adjacent contrast.
        estimates=[]
        for k in range(8):
            ix=rng.permutation(n);s=np.sign(x[ix[::2]]-x[ix[1::2]])
            estimates.append(np.mean(s[:,0]*(s[:,1]-s[:,2])))
        rows.append(dict(rep=rep,full_U=tau[0]-tau[1],matching=estimates[0],mean8=np.mean(estimates)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'variance_replications.csv',index=False)
    v=df[['full_U','matching','mean8']].var(ddof=1)
    (OUT/'variance_summary.json').write_text(json.dumps(dict(n=n,reps=replications,seed=281028903,variance=v.to_dict(),ratio_to_full=(v/v.full_U).to_dict()),indent=2))
    print(v.to_dict(),flush=True)


def timings():
    cells=revision_design();rows=[]
    for ci in [2,5,19]:
        c=cells[ci]
        for rep in range(20):
            seed,xs,_=data_for(c,200000+rep);algo=seed+20261028
            functions=[('Hoeffding',lambda:box_compare(xs)),
                       ('Local-McNemar',lambda:compare_legacy(xs,seed=algo)),
                       ('Local-E64',lambda:compare_legacy(xs,seed=algo,evidence=True,repeats=64)),
                       ('One-split',lambda:from_plans(make_plans(xs,seed=algo,K=1))),
                       ('Five-splits',lambda:from_plans(make_plans(xs,seed=algo,K=5))),
                       ('Quantile-5',lambda:from_plans(make_plans(xs,seed=algo,K=5),quantile=True)),
                       ('Pilot-E8',lambda:pilot_evidence(xs,make_plans(xs,seed=algo,K=1)[0],seed=algo)),
                       ('Multiplier',lambda:multiplier_compare(xs,seed=algo))]
            for name,fn in functions:
                start=time.perf_counter();fn();elapsed=time.perf_counter()-start
                rows.append(dict(cell=ci,p=c['p'],n=c['n'],rep=rep,method=name,seconds=elapsed))
        print('timing dimension',c['p'],'complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'standalone_timings.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['wine','variance','timings']);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);globals()[a.mode]()
