"""Honest pilot-tree selection and exact paired-concordance order tests.

Within each sample rows must be IID. Pairing is disjoint and independent of
values; the pilot and validation rows never overlap. No time-series guarantee
is claimed. Classical exact McNemar/binomial tests and Holm adjustment are
used; the graph optimizer is classical matroid intersection.
"""
from collections import deque
import numpy as np
from scipy.stats import binom
from finite_calibration import kendall_tau_a
from network_bounds import complete_edges, mst, turnover_lower_bound, turnover_upper_bound, DSU, forest, components


def fundamental_comparisons(p, edges, tree):
    adj = [[] for _ in range(p)]
    for f in sorted(tree):
        a,b=edges[f]; adj[a].append((b,f)); adj[b].append((a,f))
    pairs=[]
    for e,(a,b) in enumerate(edges):
        if e in tree: continue
        parent={a:(None,None)}; queue=deque([a])
        while b not in parent:
            v=queue.popleft()
            for u,f in adj[v]:
                if u not in parent: parent[u]=(v,f);queue.append(u)
        v=b
        while v!=a:
            u,f=parent[v];pairs.append((f,e));v=u
    return np.asarray(pairs,dtype=int).reshape(-1,2)


def holm(pvalues, alpha):
    pvalues=np.asarray(pvalues,float); m=len(pvalues)
    order=np.argsort(pvalues,kind='stable'); out=np.zeros(m,bool)
    for rank,i in enumerate(order):
        if pvalues[i] > alpha/(m-rank): break
        out[i]=True
    return out


def order_plan(x, pilot_fraction=.2, seed=1):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<10 or x.shape[1]<2 or not np.isfinite(x).all():
        raise ValueError('Expected finite n-by-p data with n >= 10 and p >= 2')
    if not 0 < pilot_fraction < 1: raise ValueError('Invalid pilot fraction')
    n,p=x.shape; rng=np.random.default_rng(seed); perm=rng.permutation(n)
    n0=max(2,int(np.floor(n*pilot_fraction)))
    if n-n0<4: raise ValueError('Need at least four validation rows')
    edges=complete_edges(p);tau=kendall_tau_a(x[perm[:n0]])
    tree=mst(p,edges,-tau); pairs=fundamental_comparisons(p,edges,tree)
    v=x[perm[n0:]]; N=len(v)//2
    signs=np.sign(v[:2*N:2]-v[1:2*N:2]).astype(np.int8)
    ends=np.asarray(edges); scores=signs[:,ends[:,0]]*signs[:,ends[:,1]]
    # A common independent U for every edge of a row pair. Ties have mean 1/2.
    u=rng.random((N,1)); bits=(u < (scores.astype(float)+1)/2)
    plus=np.zeros(len(pairs),int);minus=np.zeros(len(pairs),int)
    for start in range(0,len(pairs),256):
        fs,es=pairs[start:start+256].T
        bf=bits[:,fs];be=bits[:,es]
        plus[start:start+256]=np.sum(bf & ~be,axis=0)
        minus[start:start+256]=np.sum(~bf & be,axis=0)
    disc=plus+minus
    pv=binom.sf(plus-1,disc,.5);pv[disc==0]=1.
    return dict(p=p,edges=edges,tree=tree,pairs=pairs,pvalues=pv,
                plus=plus,minus=minus,N=N,pilot_n=n0,seed=seed,
                pilot_indices=perm[:n0],validation_indices=perm[n0:n0+2*N])


def envelope_from_rejections(plan,rejected):
    p=plan['p'];m=len(plan['edges']);tree=plan['tree'];pairs=plan['pairs']
    # A tree edge is forced if it wins against every other edge of its cut.
    # A non-tree edge is excluded if every edge of its tree path wins.
    failures=pairs[~np.asarray(rejected,bool)]
    unforced=set(failures[:,0]);unexcluded=set(failures[:,1])
    forced=set(tree)-unforced
    possible=set(tree)|unexcluded
    return forced,possible


def compare(samples, alpha=.05, pilot_fraction=.2, seed=1, upper=True):
    if len(samples)!=2 or samples[0].shape[1]!=samples[1].shape[1]:
        raise ValueError('Need two samples with identical variable dimensions')
    plans=[order_plan(x,pilot_fraction,seed+104729*s) for s,x in enumerate(samples)]
    counts=[len(z['pvalues']) for z in plans]
    rejected=holm(np.concatenate([z['pvalues'] for z in plans]),alpha)
    envs=[];start=0
    for z,M in zip(plans,counts):
        z['rejected']=rejected[start:start+M];start+=M
        envs.append(envelope_from_rejections(z,z['rejected']))
    p=plans[0]['p'];edges=plans[0]['edges']
    lower=turnover_lower_bound(p,edges,*envs)
    hi=turnover_upper_bound(p,edges,*envs) if upper else None
    return dict(lower=lower,upper=hi,envelopes=envs,plans=plans,M=sum(counts),
                rejected=int(rejected.sum()))


def all_order_plan(x,seed=1):
    x=np.asarray(x,float);n,p=x.shape;edges=complete_edges(p);m=len(edges)
    rng=np.random.default_rng(seed);idx=rng.permutation(n)[:2*(n//2)];N=n//2
    signs=np.sign(x[idx[::2]]-x[idx[1::2]]).astype(np.int8);ends=np.asarray(edges)
    scores=signs[:,ends[:,0]]*signs[:,ends[:,1]]
    bits=(rng.random((N,1))<(scores.astype(float)+1)/2).astype(float)
    both=np.rint(bits.T@bits).astype(int);ones=bits.sum(axis=0).astype(int)
    a,b=np.triu_indices(m,1)
    plus=ones[a]-both[a,b];minus=ones[b]-both[a,b];D=plus+minus
    # Both directions are fixed hypotheses, even though only one can win.
    pairs=np.concatenate([np.stack([a,b],axis=1),np.stack([b,a],axis=1)])
    plus,minus=np.r_[plus,minus],np.r_[minus,plus];D=np.r_[D,D]
    pv=binom.sf(plus-1,D,.5);pv[D==0]=1.
    return dict(p=p,edges=edges,pairs=pairs,pvalues=pv,plus=plus,minus=minus,N=N,
                tree=None,seed=seed,pilot_n=0)


def all_order_envelope(plan,rejected):
    p=plan['p'];edges=plan['edges'];m=len(edges)
    if not np.any(rejected):return set(),set(range(m))
    better=np.zeros((m,m),bool)
    rel=plan['pairs'][rejected];better[rel[:,0],rel[:,1]]=True
    F=set();P=set()
    for e,(a,b) in enumerate(edges):
        d=DSU(p)
        for f in np.flatnonzero(better[:,e]):d.union(*edges[f])
        if d.find(a)!=d.find(b):P.add(e)
        d=DSU(p)
        for f in np.flatnonzero(~better[e,:]):
            if f!=e:d.union(*edges[f])
        if d.find(a)!=d.find(b):F.add(e)
    return F,P


def compare_all(samples,alpha=.05,seed=1,upper=True):
    plans=[all_order_plan(x,seed+104729*s) for s,x in enumerate(samples)]
    counts=[len(z['pvalues']) for z in plans]
    rejected=holm(np.concatenate([z['pvalues'] for z in plans]),alpha)
    envs=[];start=0
    for z,M in zip(plans,counts):
        z['rejected']=rejected[start:start+M];start+=M
        envs.append(all_order_envelope(z,z['rejected']))
    p=plans[0]['p'];edges=plans[0]['edges']
    return dict(lower=turnover_lower_bound(p,edges,*envs),
                upper=turnover_upper_bound(p,edges,*envs) if upper else None,
                envelopes=envs,plans=plans,M=sum(counts),rejected=int(rejected.sum()))


def compare_multi(samples,alpha=.05,seed=1,repeats=5,pilot_fraction=.2,upper=True):
    if repeats<1:raise ValueError('Positive fixed number of splits required')
    parts=[compare(samples,alpha/repeats,pilot_fraction,seed+1000003*k,upper=False)
           for k in range(repeats)]
    p=samples[0].shape[1];edges=complete_edges(p);m=len(edges);envs=[]
    for s in range(2):
        F=set().union(*(r['envelopes'][s][0] for r in parts))
        P=set.intersection(*(r['envelopes'][s][1] for r in parts))
        envs.append((F,P))
    fallback=any(not F<=P or not forest(p,edges,F) or components(p,edges,P)!=1 for F,P in envs)
    if fallback:envs=[(set(),set(range(m))) for _ in range(2)]
    return dict(lower=turnover_lower_bound(p,edges,*envs),
                upper=turnover_upper_bound(p,edges,*envs) if upper else None,
                envelopes=envs,parts=parts,M=sum(r['M'] for r in parts),
                rejected=sum(r['rejected'] for r in parts),fallback=fallback)
