"""Additional deterministic hierarchy outputs and timed p=49 computations."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits
from calibration import kendall_scores_chunked
from contrast_inference import contrast_bands,contrast_envelope
from batch_calibration import batch_t_bands
from finite_calibration import finite_band
from network_bounds import complete_edges,interval_envelope,turnover_lower_bound,turnover_upper_bound,degree_bounds
from referee_revision_study import minimax,hierarchy_interval


def main(root):
    out=root/'results/empirical';p=49;edges=complete_edges(p);times={};start=time.perf_counter()
    xs=[pd.read_csv(out/f'input_period_{s}.csv',index_col=0).to_numpy() for s in [1,2]]
    fitted=[kendall_scores_chunked(x) for x in xs];times['scores_both_seconds']=time.perf_counter()-start
    tau=[x[0] for x in fitted];scores=[x[1] for x in fitted]
    existing=pd.read_csv(out/'edge_intervals.csv')
    assert np.allclose(tau[0],existing.tau1) and np.allclose(tau[1],existing.tau2)
    hierarchy_plugin=float(np.abs(minimax(p,edges,1-tau[0])-minimax(p,edges,1-tau[1])).max())
    rows=[]
    bandsets={
        'Box c=2':[(existing[f'lower{s}'].to_numpy(),existing[f'upper{s}'].to_numpy()) for s in [1,2]],
        'Finite IID sensitivity':[(b['lower'],b['upper']) for b in [finite_band(x) for x in xs]],
        'Finite unknown dependence':[(-np.ones(1176),np.ones(1176)) for _ in xs],
        'Eight-batch t':[(np.maximum(-1,b['centre']-b['halfwidth']),np.minimum(1,b['centre']+b['halfwidth'])) for b in batch_t_bands(xs)]}
    for name,bands in bandsets.items():
        lo=[1-b[1] for b in bands];hi=[1-b[0] for b in bands]
        s=time.perf_counter();envs=[interval_envelope(p,edges,a,b) for a,b in zip(lo,hi)]
        times[name+'_envelopes_seconds']=time.perf_counter()-s
        s=time.perf_counter();lh=turnover_lower_bound(p,edges,*envs);times[name+'_lower_seconds']=time.perf_counter()-s
        s=time.perf_counter();uh=turnover_upper_bound(p,edges,*envs);times[name+'_upper_seconds']=time.perf_counter()-s
        ul,uu=hierarchy_interval(p,edges,lo,hi)
        deg=[degree_bounds(p,edges,*env) for env in envs]
        width=float(np.mean(deg[1][1]-deg[0][0]-deg[1][0]+deg[0][1]))
        rows.append(dict(method=name,lower=lh,upper=uh,hierarchy_plugin=hierarchy_plugin,hierarchy_lower=ul,hierarchy_upper=uu,
            forced1=len(envs[0][0]),forced2=len(envs[1][0]),possible1=len(envs[0][1]),possible2=len(envs[1][1]),degree_width=width))
        print(name,rows[-1],flush=True)
    s=time.perf_counter();cb=contrast_bands(scores,draws=1999,rng=np.random.default_rng(20261105));times['contrast_calibration_seconds']=time.perf_counter()-s
    s=time.perf_counter();envs=[contrast_envelope(p,edges,a,b['halfwidth']) for a,b in zip(tau,cb)];times['contrast_envelopes_seconds']=time.perf_counter()-s
    old=json.loads((out/'contrast_exploration.json').read_text())
    assert [[sorted(a),sorted(b)] for a,b in envs]==old['envelopes']
    s=time.perf_counter();up=turnover_upper_bound(p,edges,*envs);times['contrast_upper_seconds']=time.perf_counter()-s
    assert up==47
    pd.DataFrame(rows).to_csv(out/'revision_summary.csv',index=False)
    iid_h=finite_band(xs[0])['halfwidth']
    record=dict(timings=times,platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,threads=1,
        timing_scope='single warm-process run; concurrent audit workload; timings are illustrative, not benchmarks',
        critical_t7=float(t.ppf(1-.05/(4*1176),7)),iid_radius=iid_h,
        hierarchy_plugin=hierarchy_plugin,contrast_upper=up,contrast_pairs=1176*1175//2)
    (out/'revision_audit.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    with threadpool_limits(1):main(Path(p.parse_args().root))
