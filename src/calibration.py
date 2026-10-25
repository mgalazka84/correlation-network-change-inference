"""Periodwise simultaneous Kendall tau-a bands for stationary time series.

Parzen lag weights and Gaussian score multipliers are standard ingredients.
The error budget is split over periods, so independence between periods is
not required for the union-bound coverage argument. The guarantee is
asymptotic for fixed dimension, within-period stationarity, geometric absolute
regularity and a positive-definite long-run score covariance;
this is not a finite-sample, high-dimensional or stationarity guarantee.
"""
import numpy as np
from scipy.stats import norm


def parzen(x):
    x = np.abs(np.asarray(x, dtype=float))
    return np.where(x <= .5, 1 - 6*x*x + 6*x**3,
                    np.where(x <= 1, 2*(1-x)**3, 0.))


def score_covariance(scores, kernel="parzen", bandwidth=None):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if bandwidth is None:
        bandwidth = max(2, int(np.ceil(2*n**(1/3))))
    if not 1 <= bandwidth < n:
        raise ValueError("Bandwidth must be between 1 and n-1")
    scores = scores - scores.mean(axis=0)
    cov = scores.T @ scores / n
    for k in range(1, bandwidth):
        weight = float(parzen(k / bandwidth)) if kernel == "parzen" else 1-k/bandwidth
        lag = scores[k:].T @ scores[:-k] / n
        cov += weight * (lag + lag.T)
    return (cov + cov.T)/2, bandwidth


def gaussian_bands(scores, alpha=.05, draws=1999, rng=None, kernel="parzen",
                   bandwidths=None, finite_df=False):
    """One max-statistic band per period, each with error alpha/len(scores).

    finite_df optionally uses a Satterthwaite t-scale sensitivity adjustment;
    it is asymptotically equivalent but has no asserted finite-sample theorem.
    The Gaussian version (finite_df=False) is the primary method.
    """
    if rng is None:
        rng = np.random.default_rng(20261026)
    kperiods = len(scores)
    if draws+1 < kperiods/alpha:
        raise ValueError("Too few multiplier draws for the requested tail probability")
    result = []
    for j, psi in enumerate(scores):
        n, m = psi.shape
        band = None if bandwidths is None else int(bandwidths[j])
        cov, band = score_covariance(psi, kernel, band)
        variance = np.diag(cov)
        if np.any(variance <= 1e-14):
            raise ValueError("Degenerate estimated score variance")
        sigma = np.sqrt(variance)
        correlation = cov / sigma[:,None] / sigma[None,:]
        eig, vectors = np.linalg.eigh(correlation)
        if eig.min() < -1e-7:
            raise AssertionError("Lag-window covariance is not positive semidefinite")
        factor = vectors * np.sqrt(np.maximum(eig,0))[None,:]
        z = rng.normal(size=(draws,m)) @ factor.T
        df = np.inf
        if finite_df:
            weights = parzen(np.arange(1,band)/band)
            df = max(3., n/(1+2*np.sum(weights**2)))
            z /= np.sqrt(rng.chisquare(df, size=draws)/df)[:,None]
        maximum = np.abs(z).max(axis=1)
        # Conservative order-statistic convention; B tends to infinity in theory.
        order = min(draws-1, int(np.ceil((draws+1)*(1-alpha/kperiods)))-1)
        critical = float(np.partition(maximum,order)[order])
        result.append({"halfwidth":critical*sigma/np.sqrt(n),"critical":critical,
                       "bandwidth":band,"df":float(df),"covariance":cov,
                       "bonferroni_halfwidth":norm.ppf(1-alpha/(2*kperiods*m))*sigma/np.sqrt(n)})
    return result


def kendall_scores_chunked(x, chunk=64):
    """Kendall tau-a and estimated projection scores, including observation ties.

    Chunking bounds temporary memory by O(chunk*n*p) instead of O(n*n*p).
    Its arithmetic is the same U-statistic calculation as the pilot code.
    """
    x = np.asarray(x,dtype=float)
    if x.ndim != 2 or len(x)<4 or not np.all(np.isfinite(x)):
        raise ValueError("A finite n-by-p array with n>=4 is required")
    n,p=x.shape
    a,b=np.triu_indices(p,1)
    projected=np.empty((n,len(a)))
    for start in range(0,n,chunk):
        stop=min(start+chunk,n)
        signs=np.sign(x[start:stop,None,:]-x[None,:,:]).astype(np.float32)
        pair=np.matmul(signs.transpose(0,2,1),signs).astype(float)/(n-1)
        projected[start:stop]=pair[:,a,b]
    tau=projected.mean(axis=0)
    return tau,2*(projected-tau)
