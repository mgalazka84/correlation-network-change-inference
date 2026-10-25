"""Reproducible, explicitly preliminary Monte Carlo experiment.

Gaussian latent AR(1) data have analytically known population Kendall weights.
The dependent Gaussian multiplier uses the Bartlett long-run covariance of
estimated first-order U-statistic scores. No finite-sample coverage is claimed.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, kendalltau, t as student_t
from threadpoolctl import threadpool_limits
from network_bounds import complete_edges, mst, degrees, compare_intervals


def covariance(p, hub, kind):
    if kind == "clear":
        rho = 0.65
        c = np.full((p,p), rho*rho)
        c[hub, :] = rho
        c[:, hub] = rho
        np.fill_diagonal(c, 1.)
        return c
    c = np.full((p,p), 0.35)
    np.fill_diagonal(c, 1.)
    if kind == "near":
        for j in range(p):
            if j != hub:
                c[hub,j] += 0.025
                c[j,hub] += 0.025
    return c


def gaussian_ar(n, cov, phi, rng):
    p = len(cov)
    chol = np.linalg.cholesky(cov)
    x = np.empty((n,p))
    x[0] = chol @ rng.normal(size=p)  # Exact stationary initial distribution.
    innovations = rng.normal(size=(n-1,p)) @ chol.T * np.sqrt(1-phi*phi)
    for i in range(1,n):
        x[i] = phi*x[i-1] + innovations[i-1]
    return x


def kendall_scores(x):
    n, p = x.shape
    signs = np.sign(x[:,None,:] - x[None,:,:]).astype(np.float32)
    h = np.matmul(signs.transpose(0,2,1), signs).astype(float) / (n-1)
    a, b = np.triu_indices(p, 1)
    projected = h[:,a,b]
    tau = projected.mean(0)
    psi = 2*(projected - tau)
    return tau, psi


def long_run_cov(psi, bandwidth):
    n = len(psi)
    cov = psi.T @ psi / n
    for k in range(1, min(bandwidth,n)):
        gamma = psi[k:].T @ psi[:-k] / n
        cov += (1-k/bandwidth)*(gamma + gamma.T)
    return (cov+cov.T)/2


def joint_halfwidths(scores, bandwidth, alpha, bootstrap_reps, rng):
    normal_max = np.zeros(bootstrap_reps)
    sigmas = []
    for psi in scores:
        cov = long_run_cov(psi, bandwidth)
        sig = np.sqrt(np.maximum(np.diag(cov), 1e-15))
        corr = cov / sig[:,None] / sig[None,:]
        values, vectors = np.linalg.eigh(corr)
        if values.min() < -1e-8:
            raise AssertionError("Bartlett covariance should be positive semidefinite")
        factor = vectors * np.sqrt(np.maximum(values,0))[None,:]
        draws = rng.normal(size=(bootstrap_reps,len(sig))) @ factor.T
        normal_max = np.maximum(normal_max, np.abs(draws).max(1))
        sigmas.append(sig)
    q = np.quantile(normal_max, 1-alpha, method="higher")
    m = scores[0].shape[1]
    qbon = norm.ppf(1-alpha/(4*m))
    widths = [q*s / np.sqrt(len(psi)) for s,psi in zip(sigmas,scores)]
    bonwidths = [qbon*s / np.sqrt(len(psi)) for s,psi in zip(sigmas,scores)]
    return widths, bonwidths, float(q)


def wilson(k, n):
    z = norm.ppf(.975)
    p = k/n
    den = 1+z*z/n
    centre = (p+z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return centre-radius, centre+radius


def validate_statistics():
    rng = np.random.default_rng(345)
    x = rng.normal(size=(40,5))
    tau, score = kendall_scores(x)
    target = [kendalltau(x[:,i],x[:,j]).statistic for i,j in complete_edges(5)]
    assert np.max(np.abs(tau-target)) < 1e-12
    assert np.max(np.abs(score.mean(0))) < 1e-12
    transformed = student_t.ppf(norm.cdf(x),3)
    tau2, score2 = kendall_scores(transformed)
    assert np.array_equal(tau,tau2) and np.array_equal(score,score2)
    return {"scipy_kendall_crosscheck":"passed", "centred_scores":"passed",
            "strictly_monotone_heavy_tail_transform":"passed",
            "note":"Transformation check is algebraic; it does not validate robustness to contamination."}


def main(args):
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    p = args.p
    edges = complete_edges(p)
    rows = []
    cases = [("clear_null", "clear",0), ("clear_change","clear",1),
             ("near_null","near",0), ("near_change","near",1),
             ("exact_tie_null","tie",0)]
    for n in args.n:
        for case_index,(case, kind, hub2) in enumerate(cases):
            c1, c2 = covariance(p,0,kind), covariance(p,hub2,kind)
            idx = np.triu_indices(p,1)
            true_tau = [2/np.pi*np.arcsin(c[idx]) for c in [c1,c2]]
            ts = [mst(p,edges,1-tau) for tau in true_tau]
            true_turnover = 0 if kind == "tie" else len(ts[0]-ts[1])
            true_delta = degrees(p,edges,ts[1])-degrees(p,edges,ts[0])
            for rep in range(args.reps):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed,n,case_index,rep]))
                samples = [gaussian_ar(n,c,args.phi,rng) for c in [c1,c2]]
                fitted = [kendall_scores(x) for x in samples]
                taus = [z[0] for z in fitted]
                scores = [z[1] for z in fitted]
                bandwidth = max(2, round(1.5*n**(1/3)))
                dep, bon, q = joint_halfwidths(scores,bandwidth,args.alpha,args.bootstrap,rng)
                iid, _, qiid = joint_halfwidths(scores,1,args.alpha,args.bootstrap,rng)
                plugin = [mst(p,edges,1-tau) for tau in taus]
                plugin_change = len(plugin[0]-plugin[1])
                for method, widths in [("dependent_multiplier",dep),
                                       ("bonferroni_hac",bon),("iid_multiplier",iid)]:
                    lowers = [np.maximum(-1,tau-h) for tau,h in zip(taus,widths)]
                    uppers = [np.minimum(1,tau+h) for tau,h in zip(taus,widths)]
                    cover = all(np.all((t>=l)&(t<=u)) for t,l,u in zip(true_tau,lowers,uppers))
                    fit = compare_intervals(p,edges,1-uppers[0],1-lowers[0],1-uppers[1],1-lowers[1])
                    dlo, dhi = fit["degree_change_lower"], fit["degree_change_upper"]
                    if kind == "tie":
                        deg_valid = np.all(dlo <= -(p-2)) and np.all(dhi >= p-2)
                    else:
                        deg_valid = np.all((dlo <= true_delta)&(true_delta <= dhi))
                    valid = bool(deg_valid and fit["turnover_lower"] <= true_turnover)
                    if cover and not valid:
                        raise AssertionError("Deterministic transfer violated on coverage event")
                    rows.append(dict(case=case,n=n,p=p,rep=rep,method=method,
                        true_turnover=true_turnover,plugin_turnover=plugin_change,
                        plugin_changed=int(plugin_change>0),box_covered=int(cover),
                        graph_valid=int(valid),turnover_lower=fit["turnover_lower"],
                        detected=int(fit["turnover_lower"]>0),
                        degree_width=float(np.mean(dhi-dlo)),
                        raw_degree_width=float(np.mean(fit["raw_degree_change_upper"]-fit["raw_degree_change_lower"])),
                        degree_changes_certified=int(np.sum((dlo>0)|(dhi<0))),
                        mean_halfwidth=float(np.mean(widths)),bandwidth=bandwidth,
                        critical=q if method=="dependent_multiplier" else qiid if method=="iid_multiplier" else np.nan))
                if (rep+1)%50 == 0:
                    print(f"{case} n={n}: {rep+1}/{args.reps}; elapsed {time.time()-start:.1f}s",flush=True)
            pd.DataFrame(rows).to_csv(outdir/"replicates.csv",index=False)
    frame = pd.DataFrame(rows)
    summary=[]
    for (case,n,method),g in frame.groupby(["case","n","method"],sort=False):
        record=dict(case=case,n=int(n),method=method,reps=len(g),
                    true_turnover=int(g.true_turnover.iloc[0]))
        for key in ["plugin_changed","box_covered","graph_valid","detected"]:
            record[key] = float(g[key].mean())
            record[key+"_wilson_low"], record[key+"_wilson_high"] = wilson(int(g[key].sum()),len(g))
        for key in ["plugin_turnover","turnover_lower","degree_width","raw_degree_width","degree_changes_certified","mean_halfwidth"]:
            record[key] = float(g[key].mean())
            record[key+"_mcse"] = float(g[key].std(ddof=1)/np.sqrt(len(g)))
        summary.append(record)
    pd.DataFrame(summary).to_csv(outdir/"summary.csv",index=False)
    metadata=dict(seed=args.seed,n=args.n,p=p,phi=args.phi,alpha=args.alpha,
                  repetitions_per_cell=args.reps,bootstrap_repetitions=args.bootstrap,
                  bandwidth="round(1.5*n**(1/3)); minimum 2",case_parameters={"clear_rho":.65,"near_base_rho":.35,"near_hub_increment":.025},
                  number_of_cells=len(args.n)*len(cases),
                  elapsed_seconds=time.time()-start,python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,
                  statistical_checks=validate_statistics(),
                  scope="Independent stationary Gaussian AR(1) regime samples; not a financial-data study.",
                  guarantee="Graph bounds are deterministic conditional on simultaneous input coverage. Bootstrap calibration is preliminary.")
    (outdir/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summary)[["case","n","method","box_covered","graph_valid","detected","turnover_lower","degree_width"]].to_string(index=False),flush=True)
    print(f"Finished in {time.time()-start:.1f}s",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reps",type=int,default=200)
    parser.add_argument("--bootstrap",type=int,default=999)
    parser.add_argument("--n",nargs="+",type=int,default=[250,1000])
    parser.add_argument("--p",type=int,default=8)
    parser.add_argument("--phi",type=float,default=.6)
    parser.add_argument("--alpha",type=float,default=.05)
    parser.add_argument("--seed",type=int,default=20261025)
    parser.add_argument("--output",default="pilot")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        main(args)
