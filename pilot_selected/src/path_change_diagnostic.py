"""Post-review explanation of four existing path cells, without new datasets.

Replays the frozen seeds and checks every Five-splits interval against the
archived output. Population diagnostics use the cut of the common forest.
This is descriptive analysis after inspection of the original results.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import binom
from run_revision_study import revision_design, data_for, population
from revision_methods import make_plans, from_plans
from network_bounds import complete_edges, DSU, components

ROOT = Path(__file__).resolve().parents[1]
CELLS = (1, 4, 25, 27)


def distances(tree, p):
    adj = [[] for _ in range(p)]
    for a, b in tree:
        adj[a].append(b); adj[b].append(a)
    D = np.zeros((p, p), dtype=int)
    for root in range(p):
        queue = [(root, -1, 0)]
        for a, parent, distance in queue:
            D[root, a] = distance
            queue += [(b, a, distance+1) for b in adj[a] if b != parent]
    return D


def geometry(c):
    p = c['p']; rho = c['rho']; edges = complete_edges(p)
    trees = [set(map(tuple, c[k])) for k in ('tree1', 'tree2')]
    removed = sorted(trees[0]-trees[1]); added = sorted(trees[1]-trees[0])
    assert len(removed) == len(added) == 1
    ds = [distances(T, p) for T in trees]
    common = trees[0] & trees[1]; dsu = DSU(p)
    for a, b in common: dsu.union(a, b)
    cross = [(a,b) for a,b in edges if dsu.find(a) != dsu.find(b)]
    lengths = [max(D[a,b] for D in ds) for a,b in cross]
    ell = min(lengths)
    tau = lambda r: float(2/np.pi*np.arcsin(r))
    return dict(cell=c['cell'], p=p, rho=rho, removed=removed, added=added,
                foreign_lengths=[int(ds[1][removed[0]]), int(ds[0][added[0]])],
                foreign_gaps=[tau(rho)-tau(rho**int(ds[1][removed[0]])),
                              tau(rho)-tau(rho**int(ds[0][added[0]]))],
                cross_edges=len(cross), cross_bottleneck_distance=int(ell),
                cross_bottleneck_gap=tau(rho)-tau(rho**int(ell)),
                minimum_gap=population(c)[6], true_M=population(c)[8])


def replay(task):
    c, rep = task
    seed, xs, pop = data_for(c, rep)
    plans = make_plans(xs, seed=seed+20261028, K=5)
    r = from_plans(plans)
    F1, P1 = r['envelopes'][0]; F2, P2 = r['envelopes'][1]
    edges = complete_edges(c['p'])
    return dict(cell=c['cell'], rep=rep, seed=seed, lower=r['lower'], upper=r['upper'],
                detect=r['lower'] > 0,
                common_possible_disconnected=components(c['p'], edges, P1 & P2) > 1,
                forced_conflict=bool((F1-P2) | (F2-P1)),
                pilot_correct=sum(z['tree']==T for split in plans
                                  for z,T in zip(split,pop[2]))/10)


def figure(details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    design = revision_design()
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2,1,figsize=(7.1,4.3))
    for ax,ci,label in zip(axes,[1,4],['(a)', '(b)']):
        c=design[ci]; d=details[str(ci)]; p=c['p']
        removed=set(map(tuple,d['removed'])); added=d['added'][0]
        for a,b in c['tree1']:
            changing=(a,b) in removed
            ax.plot([a,b],[0,0],color='#B43C34' if changing else '#555555',
                    lw=2.5,ls='--' if changing else '-',zorder=1)
        ax.add_patch(FancyArrowPatch((added[0],0),(added[1],0),
                     connectionstyle='arc3,rad=-.25',arrowstyle='-',
                     linewidth=2.5,color='#166D9A',zorder=1))
        ax.scatter(range(p),np.zeros(p),s=42,facecolor='white',edgecolor='black',zorder=2)
        for i in range(p): ax.text(i,-.22,str(i),ha='center',va='top')
        ax.set_title(f'{label} p = {p}: replace {tuple(d["removed"][0])} with {tuple(added)}',loc='left',pad=12)
        ax.text(.01,.98,f'Foreign-edge path length: {d["foreign_lengths"][0]}; cut bottleneck distance: {d["cross_bottleneck_distance"]}',
                transform=ax.transAxes,va='top',fontsize=9)
        ax.set_xlim(-.6,p-.4); ax.set_ylim(-.5,2.4 if p==16 else 1.3); ax.axis('off')
    fig.legend(handles=[Line2D([0],[0],color='#555555',lw=2,label='Common edge'),
                        Line2D([0],[0],color='#B43C34',lw=2,ls='--',label='Removed edge'),
                        Line2D([0],[0],color='#166D9A',lw=2,label='Added edge')],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1),h_pad=1.0)
    for ext in ('pdf','png'):
        fig.savefig(ROOT/'figures'/f'path_change_geometry.{ext}',dpi=170)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args(); design=revision_design(); out=ROOT/'revision_results'
    tasks=[(design[i],r) for i in CELLS for r in range(250)]
    with ProcessPoolExecutor(args.workers) as ex: records=list(ex.map(replay,tasks))
    replayed=pd.DataFrame(records)
    old=pd.read_csv(out/'expanded_replications.csv')
    old=old[old.cell.isin(CELLS)&(old.method=='Five-splits')]
    check=replayed.merge(old,on=['cell','rep'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==1000
    for name in ['seed','lower','upper','detect']:
        assert (check[name+'_new']==check[name+'_old']).all(), name
    details={}
    for i in CELLS:
        d=geometry(design[i]); rows=replayed[replayed.cell==i]
        for name in ['detect','common_possible_disconnected','forced_conflict','pilot_correct']:
            d[name]=float(rows[name].mean())
        d['replications']=len(rows); details[str(i)]=d
    replayed.to_csv(out/'path_diagnostic_replications.csv',index=False)
    result=dict(status='Post-review descriptive replay of four unchanged frozen cells',
                interval_replay_checks=1000,interval_replay_passed=True,cells=details,
                binomial_tail_15_of_250_at_point05=float(binom.sf(14,250,.05)))
    (out/'path_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    figure(details)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
