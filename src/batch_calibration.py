"""Fixed-number batch-t bands: a conservative alternative input calibration.

This uses the standard grouped-t principle, with a fixed number of contiguous
equal-size batches. Its validity is asymptotic as each batch grows under an
appropriate joint block CLT, not exact for finite dependent observations.
"""
import numpy as np
from scipy.stats import t
from calibration import kendall_scores_chunked


def batch_t_bands(samples,groups=8,alpha=.05):
    result=[]
    for x in samples:
        n,p=x.shape;m=p*(p-1)//2
        size=n//groups
        if groups<2 or size<4:raise ValueError("Insufficient observations per batch")
        estimates=np.array([kendall_scores_chunked(x[j*size:(j+1)*size])[0] for j in range(groups)])
        centre=estimates.mean(axis=0)
        se=estimates.std(axis=0,ddof=1)/np.sqrt(groups)
        if np.any(se<=1e-14):raise ValueError("Degenerate between-batch variance")
        critical=float(t.ppf(1-alpha/(2*len(samples)*m),groups-1))
        result.append(dict(centre=centre,halfwidth=critical*se,critical=critical,
                           batch_size=size,groups=groups,used=groups*size))
    return result
