"""Native vector scientific figures for the extended comparison study."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paired_order import fundamental_comparisons
from network_bounds import complete_edges
from run_revision_study import revision_design

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)


def geometry():
    cells=revision_design();fig=plt.figure(figsize=(7.6,5.0));gs=fig.add_gridspec(2,4,height_ratios=[1,1.5],hspace=.5,wspace=.8)
    for j,(ci,title) in enumerate([(13,'Star'),(0,'Path'),(6,'Random tree'),(22,'Wine topology')]):
        c=cells[ci];p=c['p'];tree=c['tree1'];edges=complete_edges(p);ids={edges.index(tuple(e)) for e in tree}
        angles=np.linspace(0,2*np.pi,p,endpoint=False);xy=np.c_[np.cos(angles),np.sin(angles)]
        if title=='Star':
            xy[0]=0;angle=np.linspace(0,2*np.pi,p-1,endpoint=False);xy[1:]=np.c_[np.cos(angle),np.sin(angle)]
        if title=='Path':xy=np.array([[-1,-.6],[-1,0],[-1,.6],[-.3,.6],[.4,.6],[1,.6],[1,0],[1,-.6]])
        ax=fig.add_subplot(gs[0,j]);ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.25,1.25);ax.set_ylim(-1.25,1.25)
        for a,b in tree:ax.plot(xy[[a,b],0],xy[[a,b],1],color='#387e98',lw=1.5)
        ax.scatter(xy[:,0],xy[:,1],s=30,facecolor='white',edgecolor='#214a5c',zorder=3)
        K=len(fundamental_comparisons(p,edges,ids))
        ax.set_title(f'{title}\n'+f'$p={p}$, $K_s={K}$',fontsize=9,pad=9)
    rho=np.linspace(.12,.98,160);tau=2/np.pi*np.arcsin(rho);delta=tau-2/np.pi*np.arcsin(rho*rho);discord=(1-tau)/2
    a=fig.add_subplot(gs[1,:2]);b=fig.add_subplot(gs[1,2:])
    a.plot(rho,delta,label=r'Weight gap $\Delta$',color='#277e8b',lw=2)
    a.plot(rho,discord,label=r'Discordance $d$',color='#bb6e38',lw=2)
    a.set_xlabel(r'Link correlation $\rho$');a.set_ylabel('Population quantity');a.legend(frameon=False,fontsize=8)
    D=np.arange(401);values=[]
    for p,M,color in [(8,84,'#277e8b'),(16,420,'#605296')]:
        critical=binom.isf(.05/(5*M),D,.5).astype(int)+1
        power=np.array([np.sum(binom.pmf(D,400,d)*binom.sf(critical-1,D,.5+gap/(4*d))) for d,gap in zip(discord,delta)])
        b.plot(rho,power,label=f'p = {p}',lw=2,color=color)
        values.extend(dict(rho=r,p=p,gap=g,discordance=d,power=z) for r,g,d,z in zip(rho,delta,discord,power))
    b.set_xlabel(r'Link correlation $\rho$');b.set_ylabel('Marginal rejection probability',fontsize=8);b.legend(frameon=False,fontsize=8)
    for ax in [a,b]:
        ax.set_xlim(.1,1);ax.set_ylim(0,.42 if ax is b else .52);ax.grid(color='#e9ecee',lw=.7)
        for value in [.35,.65]:ax.axvline(value,color='#92999d',ls=':',lw=.8)
    pd.DataFrame(values).to_csv(ROOT/'revision_results/power_curve.csv',index=False)
    save(fig,'geometry_power')


def wine():
    df=pd.read_csv(ROOT/'revision_results/wine_interval_frequencies.csv')
    intervals=sorted(set(zip(df.lower,df.upper)));methods=['One-split','Five-splits','Quantile-5','Pilot-E8']
    fig,axes=plt.subplots(1,2,figsize=(7.6,4.3),sharey=True)
    for ax,policy in zip(axes,['All records','Unique feature rows']):
        sub=df[df.policy==policy]
        for row in sub.itertuples():
            x=methods.index(row.method);y=intervals.index((row.lower,row.upper));v=row.count
            ax.scatter(x,y,s=70*v,facecolor='#277e8b',edgecolor='white',lw=.7,zorder=3)
            ax.text(x,y,str(v),ha='center',va='center',color='white',fontsize=8,zorder=4)
        ax.set_xticks(range(4),['One\nsplit','Five\nsplits','Quantile\n5','Pilot\nE8'],fontsize=8)
        ax.set_xlim(-.6,3.6);ax.set_ylim(-.6,len(intervals)-.4);ax.set_title(policy,fontsize=10)
        ax.grid(color='#e7ecef',lw=.65);ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(intervals)),[f'[{a}, {b}]' for a,b in intervals]);axes[0].set_ylabel('Reported replacement interval')
    axes[0].invert_yaxis();fig.tight_layout(w_pad=2);save(fig,'wine_seed_distribution')


def tuning():
    df=pd.read_csv(ROOT/'revision_results/tuning_replications.csv',low_memory=False)
    config=json.loads((ROOT/'revision_results/revision_protocol.json').read_text());cells=revision_design()
    fig,axes=plt.subplots(2,3,figsize=(7.6,5.3),sharex=True,sharey=True)
    for ax,ci in zip(axes.flat,config['tuning_cells']):
        c=cells[ci];s=df[df.cell==ci].groupby(['gamma','K']).detect.mean().unstack()
        assert len(df[df.cell==ci])==250*20
        ax.imshow(s.to_numpy(),origin='upper',vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for i in range(5):
            for j in range(4):
                v=s.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v>.55 else '#172e3c')
        shape='Binary star' if c['model']=='Binary-Gaussian' else c['shape'].replace('wine-topology','Wine topology').capitalize()
        ax.set_title(f'{shape}, p = {c["p"]}, H = {c["changes"]}\n'+r'$\rho$'+f' = {c["rho"]}',fontsize=8.5)
        ax.set_xticks(range(4),[1,3,5,10]);ax.set_yticks(range(5),['.1','.2','.3','.4','.5'])
    for ax in axes[1]:ax.set_xlabel('Number of splits K')
    for ax in axes[:,0]:ax.set_ylabel(r'Pilot fraction $\gamma$')
    fig.tight_layout(w_pad=1.5,h_pad=2);save(fig,'tuning_sensitivity')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['geometry','wine','tuning']);a=ap.parse_args();globals()[a.mode]()
