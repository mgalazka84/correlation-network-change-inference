"""Reproduce the exploratory wine example, including all sensitivity outputs."""
from pathlib import Path
import io,zipfile,json,hashlib,time,platform,sys
import urllib.request
import numpy as np
import pandas as pd
import scipy
from paired_order import compare,compare_all,compare_multi
from legacy_comparators import compare_legacy
from run_study import box_compare
from finite_calibration import kendall_tau_a
from network_bounds import mst,complete_edges

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'results').mkdir(parents=True,exist_ok=True)
URL='https://archive.ics.uci.edu/static/public/186/wine+quality.zip'
SEED=20261028

def load_data():
    path=ROOT/'data'/'wine_quality.zip';path.parent.mkdir(exist_ok=True)
    if not path.exists():path.write_bytes(urllib.request.urlopen(URL).read())
    with zipfile.ZipFile(path) as z:
        dfs=[pd.read_csv(z.open('winequality-'+color+'.csv'),sep=';') for color in ['red','white']]
    return dfs,path

def main():
    dfs,path=load_data();cols=list(dfs[0].columns[:-1]);edges=complete_edges(len(cols));rows=[];details=[]
    methods=[('Hoeffding',box_compare),('All-pairs',compare_all),('One-split',compare),('Five-splits',compare_multi),('Local-McNemar',lambda x,seed:compare_legacy(x,seed=seed)),('Local-E8',lambda x,seed:compare_legacy(x,seed=seed,evidence=True)),('Local-E64',lambda x,seed:compare_legacy(x,seed=seed,evidence=True,repeats=64))]
    for policy in ['All records','Unique feature rows']:
        xs=[d[cols].to_numpy() if policy=='All records' else d[cols].drop_duplicates().to_numpy() for d in dfs]
        Ts=[mst(len(cols),edges,-kendall_tau_a(x)) for x in xs]
        for label,fn in methods:
            t=time.perf_counter();r=fn(xs) if label=='Hoeffding' else fn(xs,seed=SEED)
            rows.append(dict(policy=policy,method=label,n_red=len(xs[0]),n_white=len(xs[1]),seed=SEED,
              lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower'],M=r['M'],
              rejected=int(r['rejected']),forced_red=len(r['envelopes'][0][0]),possible_red=len(r['envelopes'][0][1]),
              forced_white=len(r['envelopes'][1][0]),possible_white=len(r['envelopes'][1][1]),
              plugin_turnover=len(Ts[0]-Ts[1]),seconds=time.perf_counter()-t))
            details.append(dict(policy=policy,method=label,envelopes=[dict(forced=sorted(F),possible=sorted(P)) for F,P in r['envelopes']],plugin_trees=[sorted(T) for T in Ts]))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'wine_results.csv',index=False)
    (ROOT/'results'/'wine_details.json').write_text(json.dumps(dict(columns=cols,edges=edges,details=details),indent=2,default=int))
    # Consecutive seeds report sensitivity, never select a most favorable output.
    sensitivities=[]
    for policy in ['All records','Unique feature rows']:
        xs=[d[cols].to_numpy() if policy=='All records' else d[cols].drop_duplicates().to_numpy() for d in dfs]
        for k in range(20):
            for label,fn in [('One-split',compare),('Five-splits',compare_multi)]:
                r=fn(xs,seed=SEED+k)
                sensitivities.append(dict(policy=policy,method=label,seed=SEED+k,lower=r['lower'],upper=r['upper'],width=r['upper']-r['lower']))
    pd.DataFrame(sensitivities).to_csv(ROOT/'results'/'wine_seed_sensitivity.csv',index=False)
    manifest=dict(source=URL,doi='10.24432/C56S3T',license='CC BY 4.0',download_date='2026-10-28',
                  archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),features=cols,quality_excluded=True,
                  development='Exploratory example used during method development; not an independent confirmatory analysis',
                  duplicate_policy='Exact duplicates over 11 features; deduplication changes weighting and does not establish IID sampling',
                  python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform())
    (ROOT/'results'/'wine_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(sensitivities).groupby(['policy','method']).agg(lower_min=('lower','min'),lower_max=('lower','max'),upper_min=('upper','min'),upper_max=('upper','max'),positive=('lower',lambda x:int((x>0).sum()))))

if __name__=='__main__':main()
