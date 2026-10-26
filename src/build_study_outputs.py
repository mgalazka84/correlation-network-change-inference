"""Generate the new study tables and figures from saved local outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';MAN=ROOT/'manuscript'
FIG.mkdir(exist_ok=True);MAN.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight'})


def texrow(values):return ' & '.join(map(str,values))+r' \\'


def main():
    a=pd.read_csv(ROOT/'results/validation_study/summary.csv')
    b=pd.read_csv(ROOT/'results/targeted_study/summary.csv')
    assert a.cell.nunique()==38 and b.cell.nunique()==15
    cols=['bartlett_pilot','parzen_gaussian','parzen_t_sensitivity','bonferroni_parzen']
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',texrow([r'$n$',r'$\phi$','Bartlett','Parzen','Parzen-$t$','Bonferroni']),r'\midrule']
    for n in [250,1000]:
        for phi in [0.,.6,.85]:
            vals=[]
            for method in cols:
                d=a[(a.n==n)&(a.phi==phi)&(a.method==method)&(a.cell<30)].box_covered
                vals.append(f'{100*d.min():.1f}--{100*d.max():.1f}')
            lines.append(texrow([n,phi,*vals]))
    lines += [r'\bottomrule',r'\end{tabular}']
    (MAN/'calibration_ranges.tex').write_text('\n'.join(lines))
    names={'parzen_box':'Coordinate bands','parzen_contrasts':'Contrast bands','batch_t_8':'Eight-batch $t$'}
    cases={"clear_null":"Clear null","clear_change":"Clear change","near_null":"Near-tie null",
           "near_change":"Near-tie change","exact_tie_null":"Exact-tie null","gap_0.10_change":"Gap 0.10 change"}
    header=texrow(['Scenario',r'$n$',r'$\phi$','Method','Input','Detect.','$L_H$','Width'])
    lines=[r'\begin{longtable}{@{}lllrllll@{}}',
           r'\caption{Complete follow-up: input coverage and detection in percent; mean turnover lower bound and degree-interval width. Each cell has 200 paired datasets.}\label{tab:followup}'+chr(92)*2,
           r'\toprule',header,r'\midrule',r'\endfirsthead',
           r'\toprule',header,r'\midrule',r'\endhead']
    for cell in sorted(b.cell.unique()):
        for j,method in enumerate(names):
            r=b[(b.cell==cell)&(b.method==method)].iloc[0]
            lines.append(texrow([cases[r['case']] if j==0 else '',int(r.n) if j==0 else '',f'{r.phi:.2f}' if j==0 else '',
                ['Box','Contrast','Batch $t$'][j],f'{100*r.box_covered:.1f}',f'{100*r.detected:.1f}',f'{r.turnover_lower:.2f}',f'{r.degree_width:.2f}'])+('*' if j<2 else ''))
        if cell!=b.cell.max():lines.append(r'\addlinespace[3pt]')
    lines+=[r'\bottomrule',r'\end{longtable}']
    (MAN/'targeted_table.tex').write_text('\n'.join(lines))
    chosen=[1,6,11,13,8,14]
    labels=['Clear\nn=250, phi=.6','Clear\nn=1000, phi=.6','Clear\nn=1000, phi=.85','Clear\nn=4000, phi=.6','Near tie\nn=1000, phi=.6','Gap .10\nn=1000, phi=.6']
    fig,axes=plt.subplots(2,1,figsize=(8.1,6.1),sharex=True)
    colors=['#306d8c','#c56e29','#588146']
    for j,(method,label,color) in enumerate(zip(names,names.values(),colors)):
        d=b[b.method==method].set_index('cell').loc[chosen]
        x=np.arange(len(chosen))+(j-1)*.17
        for ax,key in zip(axes,['box_covered','detected']):
            y=d[key].to_numpy();lo=d[key+'_low'].to_numpy();hi=d[key+'_high'].to_numpy()
            ax.errorbar(x,y,yerr=np.maximum(np.vstack([y-lo,hi-y]),0),fmt='o',capsize=2,ms=4,color=color,label=label)
    axes[0].axhline(.95,color='#59636c',ls='--',lw=.8)
    axes[0].set_ylabel('Simultaneous input coverage');axes[0].set_ylim(.6,1.02)
    axes[1].set_ylabel('Positive turnover lower bound');axes[1].set_ylim(-.03,1.05)
    axes[1].set_xticks(range(len(chosen)),labels,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.18)
    handles,labels_=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels_,loc='lower center',ncol=3,bbox_to_anchor=(.5,.002),frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'followup_calibration.pdf');fig.savefig(FIG/'followup_calibration.png',dpi=160);plt.close(fig)
    eroot=ROOT/'results/empirical'
    e=pd.read_csv(eroot/'summary.csv')
    contrast=json.loads((eroot/'contrast_exploration.json').read_text())
    batch=json.loads((eroot/'batch_summary.json').read_text())
    lo=np.array(contrast['degree_change_lower']);hi=np.array(contrast['degree_change_upper'])
    blo=np.array(batch['degree_change_lower']);bhi=np.array(batch['degree_change_upper'])
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',texrow(['Calibration','Plug-in $H$','$L_H$','Mean width','Forced 1/2','Possible 1/2']),r'\midrule']
    for _,r in e.iterrows():
        lines.append(texrow([f"Box, $c={int(r.bandwidth_coefficient)}$",int(r.plugin_turnover),int(r.turnover_lower),f'{r.degree_width:.2f}',f'{int(r.mandatory1)}/{int(r.mandatory2)}',f'{int(r.possible1)}/{int(r.possible2)}']))
    env=contrast['envelopes']
    lines.append(texrow(['Contrasts, $c=2$',int(e.plugin_turnover.iloc[0]),contrast['turnover_lower'],f'{np.mean(hi-lo):.2f}',f'{len(env[0][0])}/{len(env[1][0])}',f'{len(env[0][1])}/{len(env[1][1])}']))
    lines.append(texrow(['Eight-batch $t$',int(e.plugin_turnover.iloc[0]),batch['turnover_lower'],f'{np.mean(bhi-blo):.2f}','--','--']))
    lines += [r'\bottomrule',r'\end{tabular}']
    (MAN/'empirical_table.tex').write_text('\n'.join(lines))
    nodes=pd.read_csv(eroot/'node_intervals.csv')
    nodes=nodes[nodes.bandwidth_coefficient==2].reset_index(drop=True)
    selected=sorted(range(len(nodes)),key=lambda i:(-abs(nodes.iloc[i].plugin_change),nodes.iloc[i].industry))[:8]
    fig,ax=plt.subplots(figsize=(8,4.7))
    for j,(lower,upper,color,label) in enumerate([(nodes.lower.to_numpy(),nodes.upper.to_numpy(),colors[0],'Coordinate bands'),(lo,hi,colors[1],'Contrast bands'),(blo,bhi,colors[2],'Eight-batch t')]):
        y=np.arange(8)+(j-1)*.2
        ax.hlines(y,lower[selected],upper[selected],color=color,lw=2,label=label)
    ax.scatter(nodes.iloc[selected].plugin_change,np.arange(8),marker='x',c='#242424',s=25,label='Plug-in degree change',zorder=3)
    ax.axvline(0,color='#7e868c',lw=.7)
    ax.set_yticks(range(8),nodes.iloc[selected].industry);ax.invert_yaxis()
    ax.set_xlabel('Change in degree');ax.grid(axis='x',alpha=.15)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.3),ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(FIG/'empirical_intervals.pdf');fig.savefig(FIG/'empirical_intervals.png',dpi=160);plt.close(fig)
    print('Generated three tables and two study figures from complete saved results.')


if __name__=='__main__':main()
