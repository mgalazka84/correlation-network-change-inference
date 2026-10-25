"""Reproduce the prespecified 49-industry illustration from public source data.

    python src/analyze_french.py

Downloads Kenneth French's daily 49-industry ZIP to a local data directory.
The value-weighted panel compares 2004-2007 with 2010-2013. These windows and
the bandwidth sensitivity grid are fixed in code, not selected by detection.
Rounded returns are analyzed using Kendall tau-a, without adding jitter.
Stationarity within each four-year period is a substantive assumption, not
verified by this script. Outputs do not establish a causal crisis effect.
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from calibration import gaussian_bands, kendall_scores_chunked
from contrast_inference import contrast_bands, compare_contrasts
from batch_calibration import batch_t_bands
from network_bounds import complete_edges, mst, degrees, compare_intervals, turnover_lower_bound

ROOT=Path(__file__).resolve().parents[1]
SOURCE="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_Daily_CSV.zip"
PERIODS=[("2004-01-01","2007-12-31"),("2010-01-01","2013-12-31")]


def read_panel(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        lines=z.read(name).decode("utf-8-sig").splitlines()
    start=next(i for i,s in enumerate(lines) if "Average Value Weighted Returns -- Daily" in s)+1
    end=start+1
    while end<len(lines) and lines[end].strip():
        end+=1
    frame=pd.read_csv(io.StringIO('\n'.join(lines[start:end])),index_col=0)
    frame.columns=frame.columns.str.strip()
    frame.index=pd.to_datetime(frame.index.astype(str),format="%Y%m%d")
    frame=frame.replace([-99.99,-999.],np.nan)
    return frame,lines[0]


def main(args):
    data=ROOT/'data';data.mkdir(exist_ok=True)
    archive=Path(args.input) if args.input else data/'49_Industry_Portfolios_Daily_CSV.zip'
    if not archive.exists():
        with urlopen(SOURCE,timeout=60) as response:
            raw=response.read()
        archive.write_bytes(raw)
    raw=archive.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if args.expected_sha256 and digest!=args.expected_sha256:
        raise ValueError("Source snapshot hash differs; do not silently substitute a new vintage")
    panel,vintage=read_panel(raw)
    selected=[panel.loc[a:b].copy() for a,b in PERIODS]
    if any(s.isna().any().any() for s in selected):
        raise ValueError("Missing observations in fixed windows; analysis stopped")
    out=ROOT/'results'/'empirical';out.mkdir(parents=True,exist_ok=True)
    rows=[];node_rows=[];edge_rows=[];start=time.time()
    p=len(panel.columns);edges=complete_edges(p)
    fitted=[]
    for s in selected:
        fitted.append(kendall_scores_chunked(s.to_numpy()))
        print(f"Kendall scores: {len(s)} by {p}",flush=True)
    taus=[f[0] for f in fitted];scores=[f[1] for f in fitted]
    plugin=[mst(p,edges,1-t) for t in taus]
    plugin_degree=degrees(p,edges,plugin[1])-degrees(p,edges,plugin[0])
    for coeff in [1.,2.,4.]:
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,int(coeff)]))
        bandwidths=[max(2,int(np.ceil(coeff*len(s)**(1/3)))) for s in scores]
        bands=gaussian_bands(scores,draws=args.draws,rng=rng,bandwidths=bandwidths)
        lo=[np.maximum(-1,t-b['halfwidth']) for t,b in zip(taus,bands)]
        hi=[np.minimum(1,t+b['halfwidth']) for t,b in zip(taus,bands)]
        fit=compare_intervals(p,edges,1-hi[0],1-lo[0],1-hi[1],1-lo[1])
        dlo,dhi=fit['degree_change_lower'],fit['degree_change_upper']
        certificate=turnover_lower_bound(p,edges,fit['envelope1'],fit['envelope2'],certificate=True)
        (out/f'certificate_bandwidth_{int(coeff)}.json').write_text(json.dumps(certificate,indent=2))
        rows.append(dict(bandwidth_coefficient=coeff,n1=len(selected[0]),n2=len(selected[1]),p=p,
            b1=bandwidths[0],b2=bandwidths[1],q1=bands[0]['critical'],q2=bands[1]['critical'],
            plugin_turnover=len(plugin[0]-plugin[1]),turnover_lower=fit['turnover_lower'],
            degree_width=float(np.mean(dhi-dlo)),
            raw_degree_width=float(np.mean(fit['raw_degree_change_upper']-fit['raw_degree_change_lower'])),
            certified_degree_changes=int(np.sum((dlo>0)|(dhi<0))),
            mandatory1=len(fit['envelope1'][0]),mandatory2=len(fit['envelope2'][0]),
            possible1=len(fit['envelope1'][1]),possible2=len(fit['envelope2'][1]),
            mean_halfwidth1=float(np.mean(bands[0]['halfwidth'])),mean_halfwidth2=float(np.mean(bands[1]['halfwidth']))))
        for v,name in enumerate(panel.columns):
            node_rows.append(dict(bandwidth_coefficient=coeff,industry=name,plugin_change=int(plugin_degree[v]),
                                 lower=int(dlo[v]),upper=int(dhi[v])))
        if coeff==2:
            for e,(a,b) in enumerate(edges):
                edge_rows.append(dict(industry_a=panel.columns[a],industry_b=panel.columns[b],
                    tau1=taus[0][e],tau2=taus[1][e],lower1=lo[0][e],upper1=hi[0][e],lower2=lo[1][e],upper2=hi[1][e],
                    plugin1=int(e in plugin[0]),plugin2=int(e in plugin[1]),
                    forced1=int(e in fit['envelope1'][0]),forced2=int(e in fit['envelope2'][0]),
                    possible1=int(e in fit['envelope1'][1]),possible2=int(e in fit['envelope2'][1])))
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(node_rows).to_csv(out/'node_intervals.csv',index=False)
    pd.DataFrame(edge_rows).to_csv(out/'edge_intervals.csv',index=False)
    # A separate contrast family: do not combine its conclusions with box bands.
    cb=contrast_bands(scores,draws=args.contrast_draws,rng=np.random.default_rng(20261105))
    cg=compare_contrasts(p,edges,taus,cb)
    record={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cg.items() if k!='envelopes'}
    record['envelopes']=[[sorted(a),sorted(b)] for a,b in cg['envelopes']]
    record['criticals']=[b['critical'] for b in cb];record['draws']=args.contrast_draws
    (out/'contrast_exploration.json').write_text(json.dumps(record,indent=2))
    batch=batch_t_bands([s.to_numpy() for s in selected],groups=8)
    blo=[np.maximum(-1,b['centre']-b['halfwidth']) for b in batch]
    bhi=[np.minimum(1,b['centre']+b['halfwidth']) for b in batch]
    bg=compare_intervals(p,edges,1-bhi[0],1-blo[0],1-bhi[1],1-blo[1])
    brecord={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in bg.items() if not k.startswith('envelope')}
    brecord['groups']=8;brecord['batch_sizes']=[b['batch_size'] for b in batch]
    brecord['observations_used']=[b['used'] for b in batch]
    (out/'batch_summary.json').write_text(json.dumps(brecord,indent=2))
    print('Contrast and batch-t summaries saved',flush=True)
    for j,s in enumerate(selected,1):
        s.to_csv(out/f'input_period_{j}.csv')
    metadata=dict(source=SOURCE,source_sha256=digest,source_vintage=vintage,periods=PERIODS,
        observations=[len(s) for s in selected],columns=list(panel.columns),draws=args.draws,seed=args.seed,
        alpha=.05,primary_bandwidth_coefficient=2.,bandwidth_sensitivity=[1.,2.,4.],
        missing_policy='Stop if any fixed-window value is missing; no deletion or imputation',
        estimator='Kendall tau-a on published rounded daily percent returns; no jitter',
        interpretation='Illustration under within-period stationarity; not causal and not a stationarity validation',
        contrast_draws=args.contrast_draws,batch_groups=8,elapsed_seconds=time.time()-start)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--draws',type=int,default=9999)
    parser.add_argument('--contrast-draws',type=int,default=1999)
    parser.add_argument('--seed',type=int,default=20261104)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
