"""Build finite-study tables, plots and empirical upper certificates locally."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network_bounds import complete_edges,turnover_upper_bound


def main(root):
    root=Path(root);summary=pd.read_csv(root/'results/finite_study/summary.csv')
    (root/'figures').mkdir(exist_ok=True);(root/'manuscript').mkdir(exist_ok=True)
    lines=[r'\begin{tabular}{rrlrrrr}',r'\toprule',
           r'$n$ & $\phi$ & Scenario & Input & Change & Close & Exact \\',r'\midrule']
    for _,r in summary.iterrows():
        label='binary ties' if r['kind']=='binary' else ('exact ties' if r['kind']=='tie' else ('clear change' if r['change'] else 'clear null'))
        values=' & '.join(f'{100*r[k]:.1f}' for k in ['covered','detected','stability_k1','exact_recovered'])
        lines.append(f"{int(r['n'])} & {r['phi']:g} & {label} & {values}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (root/'manuscript/finite_table.tex').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.5))
    for ax,change,metric,title in zip(axes,[True,False],['detected','stability_k1'],['Compulsory change: clear alternative','Closeness within 1 edge: clear null']):
        f=summary[(summary.phi==0)&(summary.kind=='clear')&(summary.change==change)].sort_values('n')
        y=f[metric].to_numpy()*100;lo=f[metric+'_lo'].to_numpy()*100;hi=f[metric+'_hi'].to_numpy()*100
        ax.errorbar(np.arange(len(f)),y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o-',capsize=4,color='#245d85')
        ax.set_xticks(np.arange(len(f)),[str(int(n)) for n in f.n]);ax.set_ylim(-3,103)
        ax.set_xlabel('Observations per period');ax.set_ylabel('Certification rate (%)');ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(root/'figures/finite_turnover.pdf');fig.savefig(root/'figures/finite_turnover.png',dpi=180);plt.close(fig)
    edges=complete_edges(49);f=pd.read_csv(root/'results/empirical/edge_intervals.csv')
    box=[(set(f.index[f['forced'+str(s)]==1]),set(f.index[f['possible'+str(s)]==1])) for s in [1,2]]
    contrast=json.loads((root/'results/empirical/contrast_exploration.json').read_text())
    for name,envs in [('box_c2',box),('contrast_c2',[(set(a),set(b)) for a,b in contrast['envelopes']])]:
        result=turnover_upper_bound(49,edges,*envs,certificate=True)
        (root/f'results/empirical/upper_{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['turnover_upper'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
