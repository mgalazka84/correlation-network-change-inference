"""Simultaneous within-period edge contrasts followed by graph projection.

Each period receives alpha/2. This is a separate calibration family from
coordinate bands; combining conclusions from both would require an additional
joint calibration. All pairs are included, with no estimated-tree selection.
"""
import numpy as np
from calibration import score_covariance
from network_bounds import DSU, forest, components, degree_bounds, degrees, turnover_lower_bound


def contrast_bands(scores,alpha=.05,draws=1999,rng=None,bandwidths=None):
    if rng is None:rng=np.random.default_rng(20261105)
    if draws+1 < len(scores)/alpha:
        raise ValueError("Too few multiplier draws for the requested tail probability")
    outputs=[]
    for period,psi in enumerate(scores):
        n,m=psi.shape
        cov,band=score_covariance(psi,bandwidth=None if bandwidths is None else bandwidths[period])
        variance=np.diag(cov)
        sd=np.sqrt(np.maximum(variance[:,None]+variance[None,:]-2*cov,0.))
        a,b=np.triu_indices(m,1)
        if np.any(sd[a,b]<=1e-10):
            raise ValueError("Degenerate contrast; reduce duplicate variables explicitly")
        eig,vec=np.linalg.eigh(cov)
        if eig.min() < -1e-7:raise AssertionError("Non-PSD covariance")
        z=rng.normal(size=(draws,m))@(vec*np.sqrt(np.maximum(eig,0))[None,:]).T
        maximum=np.zeros(draws)
        for start in range(0,len(a),512):
            ia,ib=a[start:start+512],b[start:start+512]
            standardized=(z[:,ia]-z[:,ib])/sd[ia,ib][None,:]
            maximum=np.maximum(maximum,np.abs(standardized).max(axis=1))
        order=min(draws-1,int(np.ceil((draws+1)*(1-alpha/len(scores))))-1)
        q=float(np.partition(maximum,order)[order])
        outputs.append(dict(halfwidth=q*sd/np.sqrt(n),critical=q,bandwidth=band))
    return outputs


def contrast_envelope(p,edges,tau,halfwidth,tol=1e-12):
    """Sound envelopes on simultaneous coverage of all tau[f]-tau[e].

    These sets need not be exact unions/intersections over the full contrast
    confidence region. The downstream optimization is exact over F subset T
    subset P, not necessarily over that smaller confidence region.
    """
    tau=np.asarray(tau);halfwidth=np.asarray(halfwidth)
    m=len(edges)
    if halfwidth.shape!=(m,m) or tau.shape!=(m,):raise ValueError("Dimensions")
    if np.any(halfwidth<0) or not np.all(np.isfinite(halfwidth)):raise ValueError("Halfwidths")
    forced,possible=set(),set()
    for e,(a,b) in enumerate(edges):
        delta=tau-tau[e]
        definite=DSU(p);potential=DSU(p)
        for f in np.flatnonzero(delta>halfwidth[:,e]+tol):
            if f!=e:definite.union(*edges[f])
        for f in np.flatnonzero(delta>=-halfwidth[:,e]-tol):
            if f!=e:potential.union(*edges[f])
        if definite.find(a)!=definite.find(b):possible.add(e)
        if potential.find(a)!=potential.find(b):forced.add(e)
    assert forced<=possible and forest(p,edges,forced) and components(p,edges,possible)==1
    return forced,possible


def compare_contrasts(p,edges,taus,bands):
    envelopes=[contrast_envelope(p,edges,tau,band['halfwidth']) for tau,band in zip(taus,bands)]
    l1,u1=degree_bounds(p,edges,*envelopes[0]);l2,u2=degree_bounds(p,edges,*envelopes[1])
    return dict(turnover_lower=turnover_lower_bound(p,edges,*envelopes),
                degree_change_lower=l2-u1,degree_change_upper=u2-l1,envelopes=envelopes,
                raw_degree_change_lower=degrees(p,edges,envelopes[1][0])-degrees(p,edges,envelopes[0][1]),
                raw_degree_change_upper=degrees(p,edges,envelopes[1][1])-degrees(p,edges,envelopes[0][0]))
