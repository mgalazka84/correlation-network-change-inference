"""Build tables and explanatory figures for the dimensional and target audits."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_explanatory_figures import tree,save,BLUE,ORANGE,GREY,DARK
from network_bounds import complete_edges


def table(path,header,rows,fmt):
    path.write_text('\n'.join([r'\begin{tabular}{'+fmt+'}',r'\toprule',header+r' \\',r'\midrule']+
        [' & '.join(map(str,row))+r' \\' for row in rows]+[r'\bottomrule',r'\end{tabular}']))


def main(root):
    m=root/'manuscript';figdir=root/'figures';r=root/'results'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    # Keep all four-node diagrams at the same A/B/C/D positions.
    edges=complete_edges(4);loc=np.array([[0,1],[1,1],[0,0],[1,0]])
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.6))
    for ax,forced,title in zip(axes,[{0,1},{4,5}],['Period 1: A has at least 2 links','Period 2: D has at least 2 links']):
        for e in [0,1,2,4,5]:
            a,b=edges[e];ax.plot(loc[[a,b],0],loc[[a,b],1],color=BLUE if e in forced else GREY,lw=3 if e in forced else 1.5,ls='-' if e in forced else '--')
        ax.scatter(loc[:,0],loc[:,1],s=330,color='white',edgecolors=DARK,zorder=4)
        for label,xy in zip('ABCD',loc):ax.text(*xy,label,ha='center',va='center',zorder=5,fontsize=10)
        ax.set_xlim(-.25,1.25);ax.set_ylim(-.25,1.25);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.text(.5,.02,'Solid: mandatory edge    Dashed: possible edge    B-C: impossible',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,figdir,'analytic_example')
    # Identical populations: graph diameter is not family distance.
    fig=plt.figure(figsize=(7.4,2.4));gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.6])
    for i,chosen in enumerate([{0,1,5},{2,3,4}]):
        ax=fig.add_subplot(gs[0,i]);tree(ax,edges,chosen,set(),BLUE if i==0 else ORANGE)
        ax.set_title('Optimal tree '+str(i+1),fontsize=10)
    ax=fig.add_subplot(gs[0,2]);ax.axis('off')
    ax.text(0,.92,'Same population in both periods',fontsize=10,weight='bold',color=DARK)
    ax.text(0,.72,'All six edge weights equal to 1',fontsize=9,color=DARK)
    ax.text(0,.48,r'$H_{\min}=0,\quad H_{\max}=3$',fontsize=13,color=DARK)
    ax.text(0,.25,r'$d_{\mathrm{H}}=0,\quad D_u=0$',fontsize=13,color=DARK)
    ax.text(0,.03,'Different links; identical family and hierarchy',fontsize=8.5,color=DARK)
    fig.tight_layout(pad=.5);save(fig,figdir,'target_comparison')
    s=pd.read_csv(r/'revision_study/summary.csv');rows=[]
    for _,v in s.iterrows():
        rows.append(['SV' if v.sv else 'IID',f'{v.strength:.2f}','Change' if v.change else 'Null',
                     f'{100*v.covered:.0f}',f'{100*v.detected:.0f}',f'[{v.mean_lower:g},{v.mean_upper:g}]',
                     f'{100*v.hierarchy_detected:.0f}',f'{100*v.batch_covered:.0f}'])
    table(m/'revision_table.tex',r'Model & $q$ & Scenario & Input & Turnover & Range & Hierarchy & Batch input',rows,'lrllrrrr')
    finite=pd.read_csv(r/'finite_study/summary.csv');finite_rows=[]
    for _,v in finite.iterrows():
        scenario='Binary' if v['kind']=='binary' else 'Ties' if v['kind']=='tie' else 'Change' if v['change'] else 'Null'
        finite_rows.append([int(v.n),f'{v.phi:g}',scenario,f'{v.halfwidth:.3f}',int(v.spacing),int(v.retained)]+
                           [f'{100*v[k]:.1f}' for k in ['covered','detected','stability_k1','exact_recovered']])
    table(m/'finite_table.tex',r'$n$ & $\phi$ & Scenario & $h$ & $a$ & $N$ & Input & Change & Pair & Exact',finite_rows,'rrlrrrrrrr')
    # Preserve nominal guarantees in labels rather than conflating empirical rates with proofs.
    change=s[s.change].reset_index(drop=True);fig,axes=plt.subplots(1,2,figsize=(7.4,2.8))
    labels=['IID, q=.70','IID, q=.94','SV, q=.94']
    for j,(metric,title) in enumerate([('detected','Compulsory turnover'),('hierarchy_detected','Hierarchy change')]):
        vals=100*change[metric].to_numpy();low=100*change[metric+'_lo'].to_numpy();high=100*change[metric+'_hi'].to_numpy()
        axes[j].errorbar(range(3),vals,yerr=np.maximum(0,np.array([vals-low,high-vals])),fmt='o',color=BLUE,capsize=4)
        axes[j].set_xticks(range(3),labels,fontsize=8);axes[j].set_ylim(-6,106);axes[j].set_yticks([0,50,100]);axes[j].set_title(title,fontsize=10)
        axes[j].set_ylabel('Detection (%)');axes[j].spines[['top','right']].set_visible(False);axes[j].grid(axis='y',alpha=.15)
    fig.tight_layout(pad=.8);save(fig,figdir,'dimensional_detection')
    e=pd.read_csv(r/'empirical/revision_summary.csv');rows=[]
    for name,label in [('Box c=2',r'Box, $c=2$'),('Eight-batch t',r'Eight-batch $t$'),('Finite IID sensitivity','Finite, IID sensitivity'),('Finite unknown dependence','Finite, unknown dependence')]:
        v=e[e.method==name].iloc[0]
        rows.append([label,int(v.lower),int(v.upper),f'{v.degree_width:.2f}',f'{v.forced1}/{v.forced2}',f'{v.possible1}/{v.possible2}'])
    old=json.loads((r/'empirical/contrast_exploration.json').read_text());env=old['envelopes'];width=np.mean(np.array(old['degree_change_upper'])-old['degree_change_lower'])
    rows.insert(1,[r'Contrasts, $c=2$',0,47,f'{width:.2f}',f'{len(env[0][0])}/{len(env[1][0])}',f'{len(env[0][1])}/{len(env[1][1])}'])
    table(m/'empirical_table.tex',r'Calibration & $L_H$ & $U_H$ & Mean width & Forced 1/2 & Possible 1/2',rows,'lrrrrr')
    print('Built revised tables and three figures; one existing diagram aligned.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(Path(p.parse_args().root))
