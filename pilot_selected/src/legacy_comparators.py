"""Companion-paper baselines, with explicit two-sample Holm control.

Formulas reproduce Sections 3 and 3.2 of Galazka and Wdowicka (2026),
immutable companion commit df81f96ef2d8ca67fae6bc34df19289f95346ce4.
They are reference implementations, not new methods of this study.
"""
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from paired_order import holm,all_order_envelope
from finite_calibration import kendall_tau_a
from network_bounds import complete_edges,turnover_lower_bound,turnover_upper_bound,forest,components


def local_pairs(edges):
    return np.asarray([(a,b) for a,e in enumerate(edges) for b,f in enumerate(edges)
                       if a!=b and bool(set(e)&set(f))],dtype=int)


def closure_pairs(pairs,rejected,m):
    better=np.zeros((m,m),bool)
    a,b=pairs[rejected].T;better[a,b]=True
    for k in range(m):better|=better[:,k:k+1]&better[k:k+1,:]
    return np.argwhere(better)


def legacy_plan(x,seed=1,evidence=False,repeats=8):
    n,p=x.shape;edges=complete_edges(p);ends=np.asarray(edges);pairs=local_pairs(edges)
    fs,es=pairs.T;rng=np.random.default_rng(seed);N=n//2;logs=[]
    fractions=np.array([1/32,1/16,1/8,1/4,1/2,3/4,min(.5,N**-.5)])
    for r in range(repeats if evidence else 1):
        idx=rng.permutation(n)[:2*N]
        sign=np.sign(x[idx[::2]]-x[idx[1::2]]).astype(np.int8)
        if not evidence:
            # Independent coordinate tie signs, equivalent to lexicographic jitter.
            sign=np.where(sign==0,np.where(rng.random(sign.shape)<.5,-1,1),sign)
        H=sign[:,ends[:,0]]*sign[:,ends[:,1]]
        pos=(H==1).astype(float);neg=(H==-1).astype(float)
        cross=np.rint(pos.T@neg).astype(int)
        plus=cross[fs,es];minus=cross[es,fs]
        if not evidence:
            D=plus+minus;pv=binom.sf(plus-1,D,.5);pv[D==0]=1.
        else:
            zero=(H==0).astype(float)
            half=np.rint(pos.T@zero+zero.T@neg).astype(int)
            counts=np.stack([plus,minus,half[fs,es],half[es,fs]],axis=1)
            logs.append(counts@np.log1p(np.array([1.,-1.,.5,-.5])[:,None]*fractions))
    if evidence:
        logE=logsumexp(np.concatenate(logs,axis=1),axis=1)-np.log(7*repeats)
        pv=np.exp(np.minimum(0.,-logE))
    return dict(p=p,edges=edges,pairs=pairs,pvalues=pv,N=N,
                tau=kendall_tau_a(x),evidence=evidence,seed=seed)


def compare_legacy(samples,alpha=.05,seed=1,evidence=False,repeats=8):
    plans=[legacy_plan(x,seed+104729*s,evidence,repeats) for s,x in enumerate(samples)]
    rejected=holm(np.concatenate([z['pvalues'] for z in plans]),alpha);envs=[];j=0
    for plan in plans:
        m=len(plan['pvalues']);keep=rejected[j:j+m].copy();j+=m
        if evidence:
            f,e=plan['pairs'].T;keep&=plan['tau'][f]>plan['tau'][e]
        plan['rejected']=keep
        rel=closure_pairs(plan['pairs'],keep,len(plan['edges']))
        temp=dict(plan,pairs=rel)
        envs.append(all_order_envelope(temp,np.ones(len(rel),bool)))
    p=plans[0]['p'];edges=plans[0]['edges']
    fallback=any(not F<=P or not forest(p,edges,F) or components(p,edges,P)!=1 for F,P in envs)
    if fallback:envs=[(set(),set(range(len(edges)))) for _ in range(2)]
    return dict(lower=turnover_lower_bound(p,edges,*envs),upper=turnover_upper_bound(p,edges,*envs),
                envelopes=envs,plans=plans,M=j,rejected=sum(z['rejected'].sum() for z in plans),fallback=fallback)
