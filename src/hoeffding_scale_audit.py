"""Monte Carlo scale comparison; no tuning or network-power experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for j,rho in enumerate([.65,.65**2]):
        rng=np.random.default_rng(2026102603+j);n=16000
        for rep in range(1000):
            z=rng.normal(size=(n,2));z[:,1]=rho*z[:,0]+np.sqrt(1-rho*rho)*z[:,1]
            rows.append(dict(rho=rho,n=n,rep=rep,tau=kendalltau(z[:,0],z[:,1]).statistic))
    f=pd.DataFrame(rows);f.to_csv(out/'replicates.csv',index=False)
    h=np.sqrt(2*np.log(2*28/.025)/(16000//2))
    s=f.groupby('rho').agg(sd=('tau','std'),mean=('tau','mean')).reset_index();s['radius']=h;s['radius_over_sd']=h/s.sd
    s.to_csv(out/'summary.csv',index=False)
    (out/'design.json').write_text(json.dumps(dict(n=16000,reps=1000,rhos=[.65,.65**2],seed=2026102603,
        seed_rule='seed plus rho index',purpose='Single-edge estimator scale only; not additional paired-network simulations'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
