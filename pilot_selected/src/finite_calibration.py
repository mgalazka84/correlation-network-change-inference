"""Finite-sample Kendall tau-a coordinate bands.

Hoeffding's bounded order-two U-statistic inequality is applied after optional
deterministic thinning. Dependence requires an externally justified upper bound
on beta(spacing) in the total-variation convention. It is NOT estimated here.
No cross-period independence is required. For IID rows use spacing=1, beta=0.
"""
import numpy as np
from scipy.stats import kendalltau


def kendall_tau_a(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite n-by-p data with n,p >= 2")
    n, p = x.shape
    pairs = n*(n-1)//2
    untied = []
    for j in range(p):
        counts = np.unique(x[:, j], return_counts=True)[1]
        untied.append(pairs - np.sum(counts*(counts-1)//2))
    result = []
    for a in range(p):
        for b in range(a+1, p):
            if untied[a] == 0 or untied[b] == 0:
                result.append(0.)
            else:
                tau_b = kendalltau(x[:, a], x[:, b]).statistic
                result.append(tau_b*np.sqrt(float(untied[a])*float(untied[b]))/pairs)
    return np.asarray(result)


def finite_band(x, alpha_period=.025, spacing=1, beta_upper=0.):
    if not 0 < alpha_period < 1 or not 0 <= beta_upper <= 1:
        raise ValueError("Invalid error budget or dependence bound")
    if not isinstance(spacing, (int, np.integer)) or spacing < 1:
        raise ValueError("Spacing must be a positive integer fixed before data")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite n-by-p data")
    x = x[::spacing]
    if len(x) < 2:
        m = x.shape[1]*(x.shape[1]-1)//2
        return dict(centre=np.zeros(m),halfwidth=2.,lower=-np.ones(m),upper=np.ones(m),
                    retained=len(x),spacing=spacing,coupling_error=0.,
                    alpha_period=alpha_period,beta_upper=beta_upper)
    tau = kendall_tau_a(x)
    n, m = len(x), len(tau)
    delta = min(1., (n-1)*beta_upper)
    remaining = alpha_period - delta
    halfwidth = min(2., np.sqrt(2*np.log(2*m/remaining)/(n//2))) if remaining > 0 else 2.
    return dict(centre=tau, halfwidth=halfwidth,
                lower=np.maximum(-1., tau-halfwidth), upper=np.minimum(1., tau+halfwidth),
                retained=n, spacing=spacing, coupling_error=delta,
                alpha_period=alpha_period, beta_upper=beta_upper)


def gaussian_ar_beta_bound(p, phi, spacing):
    """Pinsker bound for a known stationary scalar-coefficient Gaussian AR(1).

    This model-based bound is for simulations, not an estimated market-data bound.
    """
    if p < 1 or not abs(phi) < 1 or spacing < 1:
        raise ValueError("Invalid Gaussian AR parameters")
    return min(1., .5*np.sqrt(-p*np.log1p(-abs(phi)**(2*spacing))))


def gaussian_ar_spacing(n, p, phi, budget=.0125):
    for spacing in range(1, n):
        retained = (n-1)//spacing + 1
        beta = gaussian_ar_beta_bound(p, phi, spacing)
        if (retained-1)*beta <= budget:
            return spacing, beta
    raise ValueError("No usable spacing")
