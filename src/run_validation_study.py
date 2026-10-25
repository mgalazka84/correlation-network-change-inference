"""Independent validation of prespecified calibration variants.

Run from the project root:
    python src/run_validation_study.py --reps 300 --draws 1999
Outputs are generated locally and are not part of the public code repository.
The original exploratory pilot is retained separately.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from calibration import gaussian_bands, kendall_scores_chunked
from run_pilot import covariance, gaussian_ar, joint_halfwidths, wilson
from network_bounds import complete_edges, mst, degrees, compare_intervals

ROOT=Path(__file__).resolve().parents[1]


def design():
    cells=[]
    for phi in [0.,.6,.85]:
        for n in [250,1000]:
            for case,kind,hub in [("clear_null","clear",0),("clear_change","clear",1),
                ("near_null","near",0),("near_change","near",1),("exact_tie_null","tie",0)]:
                cells.append(dict(n=n,p=8,phi=phi,case=case,kind=kind,hub=hub,increment=None))
    for inc in [.05,.10,.15,.20]:
        for hub in [0,1]:
            cells.append(dict(n=1000,p=8,phi=.6,case=f"gap_{inc:.2f}_{'null' if hub==0 else 'change'}",
                              kind="gap",hub=hub,increment=inc))
    return cells


def make_cov(p,hub,kind,increment):
    if kind!="gap":
        return covariance(p,hub,kind)
    c=np.full((p,p),.35);np.fill_diagonal(c,1.)
    c[hub,:]+=(np.arange(p)!=hub)*increment
    c[:,hub]+=(np.arange(p)!=hub)*increment
    if np.linalg.eigvalsh(c).min()<=0:
        raise ValueError("Non-positive definite design")
    return c


def summarize(frame):
    records=[]
    keys=["cell","case","n","p","phi","method"]
    for key,g in frame.groupby(keys,sort=False):
        row=dict(zip(keys,key));row["reps"]=len(g)
        row["true_turnover"]=int(g.true_turnover.iloc[0])
        for name in ["box_covered","graph_valid","detected","plugin_changed"]:
            row[name]=float(g[name].mean())
            lo,hi=wilson(int(g[name].sum()),len(g))
            row[name+"_low"]=float(lo);row[name+"_high"]=float(hi)
        for name in ["turnover_lower","degree_width","degree_changes_certified","mean_halfwidth","plugin_turnover","raw_degree_width"]:
            row[name]=float(g[name].mean())
            row[name+"_mcse"]=float(g[name].std(ddof=1)/np.sqrt(len(g)))
        records.append(row)
    return pd.DataFrame(records)


def main(args):
    out=ROOT/"results"/args.output;out.mkdir(parents=True,exist_ok=True)
    cells=design()
    if args.cells is not None:
        cells=[cells[j] for j in args.cells]
    method_names=["parzen_gaussian","parzen_t_sensitivity","bonferroni_parzen","bartlett_pilot"]
    meta={"seed":args.seed,"reps":args.reps,"draws":args.draws,"alpha":.05,
          "cells":cells,"methods":method_names,"bandwidth":"ceil(2*n^(1/3)); Parzen kernel",
          "t_adjustment":"Sensitivity only: df=n/(1+2*sum(kernel_weights^2)); no finite-sample theorem",
          "primary":"parzen_gaussian","python":platform.python_version(),"numpy":np.__version__,
          "scipy":scipy.__version__,"pandas":pd.__version__,
          "design_status":"Written before new validation draws; motivated by the separate original pilot.",
          "scope":"Fixed p=8; stationary Gaussian AR(1); independent periods in simulation only."}
    (out/"design.json").write_text(json.dumps(meta,indent=2))
    start=time.time();rows=[]
    for cell_index,cell in enumerate(cells):
        n,p,phi=cell["n"],cell["p"],cell["phi"]
        edges=complete_edges(p);idx=np.triu_indices(p,1)
        covs=[make_cov(p,h,cell["kind"],cell["increment"]) for h in [0,cell["hub"]]]
        true_tau=[2/np.pi*np.arcsin(c[idx]) for c in covs]
        trees=[mst(p,edges,1-t) for t in true_tau]
        true_h=0 if cell["kind"]=="tie" else len(trees[0]-trees[1])
        true_degree=degrees(p,edges,trees[1])-degrees(p,edges,trees[0])
        for rep in range(args.reps):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,cell_index,rep]))
            xs=[gaussian_ar(n,c,phi,rng) for c in covs]
            fits=[kendall_scores_chunked(x) for x in xs]
            taus=[f[0] for f in fits];scores=[f[1] for f in fits]
            normal=gaussian_bands(scores,draws=args.draws,rng=rng)
            tscale=gaussian_bands(scores,draws=args.draws,rng=rng,finite_df=True)
            old,_,_=joint_halfwidths(scores,max(2,round(1.5*n**(1/3))),.05,args.draws,rng)
            widths=[[a["halfwidth"] for a in normal],[a["halfwidth"] for a in tscale],
                    [a["bonferroni_halfwidth"] for a in normal],old]
            plugin=[mst(p,edges,1-tau) for tau in taus]
            plugin_h=len(plugin[0]-plugin[1])
            for method,hw in zip(method_names,widths):
                lower=[np.maximum(-1,t-h) for t,h in zip(taus,hw)]
                upper=[np.minimum(1,t+h) for t,h in zip(taus,hw)]
                cover=all(np.all((t>=l)&(t<=u)) for t,l,u in zip(true_tau,lower,upper))
                graph=compare_intervals(p,edges,1-upper[0],1-lower[0],1-upper[1],1-lower[1])
                dlo,dhi=graph["degree_change_lower"],graph["degree_change_upper"]
                good_degree=(np.all(dlo<=-(p-2)) and np.all(dhi>=p-2)) if cell["kind"]=="tie" else np.all((dlo<=true_degree)&(true_degree<=dhi))
                valid=bool(good_degree and graph["turnover_lower"]<=true_h)
                assert not cover or valid
                rows.append(dict(cell=cell_index,case=cell["case"],n=n,p=p,phi=phi,rep=rep,
                    method=method,true_turnover=true_h,box_covered=int(cover),graph_valid=int(valid),
                    detected=int(graph["turnover_lower"]>0),turnover_lower=graph["turnover_lower"],
                    degree_width=float(np.mean(dhi-dlo)),
                    raw_degree_width=float(np.mean(graph["raw_degree_change_upper"]-graph["raw_degree_change_lower"])),
                    degree_changes_certified=int(np.sum((dlo>0)|(dhi<0))),mean_halfwidth=float(np.mean(hw)),
                    plugin_changed=int(plugin_h>0),plugin_turnover=plugin_h))
        frame=pd.DataFrame(rows)
        frame.to_csv(out/"replicates.csv",index=False)
        summarize(frame).to_csv(out/"summary.csv",index=False)
        print(f"cell {cell_index+1}/{len(cells)} {cell['case']} n={n} phi={phi}: {args.reps} reps; {time.time()-start:.1f}s",flush=True)
    meta["elapsed_seconds"]=time.time()-start
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reps",type=int,default=300)
    parser.add_argument("--draws",type=int,default=1999)
    parser.add_argument("--seed",type=int,default=20261103)
    parser.add_argument("--output",default="validation_study")
    parser.add_argument("--cells",type=int,nargs="+")
    with threadpool_limits(limits=1):
        main(parser.parse_args())
