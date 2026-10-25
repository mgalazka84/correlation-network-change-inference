"""Numerical checks independent of the Monte Carlo coverage experiments."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from calibration import parzen,score_covariance,kendall_scores_chunked
from run_pilot import kendall_scores
from contrast_inference import contrast_envelope
from network_bounds import complete_edges,interval_envelope


def main():
    rng=np.random.default_rng(9187);checks=[]
    for n,p in [(7,3),(38,5),(64,8)]:
        for tied in [False,True]:
            x=rng.normal(size=(n,p))
            if tied:x=np.round(x)
            a,b=kendall_scores(x);c,d=kendall_scores_chunked(x,chunk=5)
            assert np.allclose(a,c,atol=1e-14) and np.allclose(b,d,atol=1e-14)
            checks.append(dict(n=n,p=p,tied=tied,score_difference=float(np.max(np.abs(b-d)))))
    for n in [20,57,100]:
        psi=rng.normal(size=(n,6));psi-=psi.mean(0);band=7
        cov,_=score_covariance(psi,bandwidth=band)
        kernel=parzen((np.arange(n)[:,None]-np.arange(n)[None,:])/band)
        assert np.linalg.eigvalsh(kernel).min()>-1e-10
        direct=psi.T@kernel@psi/n
        assert np.allclose(cov,direct,atol=1e-12)
        checks.append(dict(n=n,psd_minimum=float(np.linalg.eigvalsh(kernel).min()),covariance_error=float(np.max(np.abs(cov-direct)))))
    count=0
    for p in [3,4,5,8]:
        edges=complete_edges(p);m=len(edges)
        for rep in range(50):
            centre=rng.integers(-4,5,size=m)/10
            radius=np.zeros(m) if rep<10 else rng.uniform(0,.15,m)
            hw=radius[:,None]+radius[None,:];np.fill_diagonal(hw,0)
            a=contrast_envelope(p,edges,centre,hw)
            b=interval_envelope(p,edges,1-centre-radius,1-centre+radius)
            assert a==b
            count+=1
    out=Path(__file__).resolve().parents[1]/'validation';out.mkdir(exist_ok=True)
    result=dict(status='passed',checks=checks,contrast_box_equivalence_checks=count,
                scope='Numerical implementation checks, not a coverage theorem')
    (out/'calibration_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
