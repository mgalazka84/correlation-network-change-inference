"""Vector figures for the pilot-selected comparison study."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
METHODS=['Hoeffding','All-pairs','Local-McNemar','Local-E8','One-split','Five-splits']
COLORS=['#9a9a9a','#4486b4','#b28c45','#83569c','#21978a','#d45b3a']
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def overview():
    fig,ax=plt.subplots(figsize=(7.7,2.1));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    pos=np.array([[.5,1.8],[1.5,2.5],[1.5,1.1],[2.5,1.8]])
    for shift,edges,col,title in [(0,[(0,1),(0,2),(1,3)],'#4486b4','Estimated group 1'),(3,[(0,1),(0,2),(2,3)],'#d45b3a','Estimated group 2')]:
        xy=pos+np.array([shift,0]);ax.text(1.5+shift,2.95,title,ha='center',fontsize=10)
        for a,b in edges:ax.plot(xy[[a,b],0],xy[[a,b],1],color=col,lw=2)
        ax.scatter(xy[:,0],xy[:,1],s=180,facecolor='white',edgecolor=col,zorder=4)
        for j,(x,y) in enumerate(xy):ax.text(x,y,str(j+1),ha='center',va='center',zorder=5,fontsize=8)
    ax.text(3,.62,'Different trees can reflect sampling error.',ha='center',fontsize=9)
    ax.text(8,2.9,'Independent selection and validation',ha='center',fontsize=10)
    for y,label in [(2.35,'Pilot tree selects comparisons'),(1.5,'Validation certifies edge orders'),(.65,'Turnover interval [L, U]')]:
        ax.add_patch(FancyBboxPatch((6.2,y-.25),3.6,.5,boxstyle='round,pad=.06',fc='#f1f4f6',ec='#688392',lw=.8))
        ax.text(8,y,label,ha='center',va='center',fontsize=8.5)
    for y in [2.05,1.2]:ax.annotate('',(8,y-.27),(8,y),arrowprops=dict(arrowstyle='->',color='#688392'))
    save(fig,'overview')

def cut_cycle():
    fig,axs=plt.subplots(1,2,figsize=(7.4,2.8));xy=np.array([[0,0],[0,1],[1.4,1],[1.4,0]])
    for ax in axs:
        ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-.4,1.8);ax.set_ylim(-.6,1.5)
        for a,b in [(0,1),(1,2),(2,3)]:ax.plot(xy[[a,b],0],xy[[a,b],1],color='#287c9b',lw=2.5)
        ax.scatter(xy[:,0],xy[:,1],s=210,c='white',edgecolor='#24424c',zorder=4)
        for j,(x,y) in enumerate(xy):ax.text(x,y,str(j+1),ha='center',va='center',zorder=5)
    axs[0].plot(xy[[0,3],0],xy[[0,3],1],ls='--',lw=2,color='#d45b3a')
    axs[0].set_title('Exclude edge (1,4) using its cycle',fontsize=10)
    axs[0].text(.7,-.33,r'All three path edges must beat (1,4).',ha='center',fontsize=8)
    for a,b in [(0,2),(0,3),(1,3)]:axs[1].plot(xy[[a,b],0],xy[[a,b],1],ls='--',lw=1.3,color='#d45b3a',alpha=.9)
    axs[1].plot([.7,.7],[-.2,1.2],ls=':',color='#555555',lw=1.2)
    axs[1].set_title('Force edge (2,3) using its cut',fontsize=10)
    axs[1].text(.7,-.33,'(2,3) must beat each dashed cut edge.',ha='center',fontsize=8)
    fig.tight_layout(w_pad=3);save(fig,'cut_cycle')

def simulations():
    df=pd.read_csv(ROOT/'results'/'combined_summary.csv')
    for metric,filename in [('detect','simulation_detection'),('width','simulation_width')]:
        fig,axs=plt.subplots(2,3,figsize=(7.4,5.6),sharex=True)
        for r,p in enumerate([8,16]):
            for c,model in enumerate(['Gaussian','Student-t5','Binary-Gaussian']):
                ax=axs[r,c];sub=df[(df.p==p)&(df.model==model)&df.change]
                settings=sorted(set(zip(sub.n,sub.rho)))
                standard=[(400,.35),(400,.65),(1000,.35),(1000,.65)]
                ys=np.array([standard.index(x) for x in settings])
                for j,(method,col) in enumerate(zip(METHODS,COLORS)):
                    sel=sub[sub.method==method].set_index(['n','rho']).loc[settings]
                    y=ys+(j-2.5)*.105
                    if metric=='detect':
                        v=sel.detect.to_numpy();lo=sel.detect_lo.to_numpy();hi=sel.detect_hi.to_numpy()
                        ax.errorbar(v,y,xerr=np.maximum(0,np.vstack([v-lo,hi-v])),fmt='o',ms=3,lw=.7,color=col,label=method)
                    else:ax.plot(sel.width/(p-1),y,'o',ms=3.6,color=col,label=method)
                ax.set_yticks(ys,[f'{int(n)}, {rho:.2f}' for n,rho in settings],fontsize=9)
                ax.set_ylim(3.65,-.65);ax.set_xlim(-.03,1.04);ax.grid(axis='x',color='#e4e8eb',lw=.6)
                title_model=r'Student $t_5$' if model=='Student-t5' else model
                ax.set_title(f'{title_model}, p = {p}',fontsize=9)
                if r==1:ax.set_xlabel('Detection probability' if metric=='detect' else 'Mean relative width',fontsize=8)
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1),w_pad=2,h_pad=2);save(fig,filename)

def wine():
    df=pd.read_csv(ROOT/'results'/'wine_results.csv');fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),sharey=True)
    methods=METHODS+(['Local-E64'] if 'Local-E64' in df.method.values else [])
    colors=COLORS+['#3b4c87']
    for ax,policy in zip(axs,['All records','Unique feature rows']):
        sub=df[df.policy==policy].set_index('method')
        for j,(method,col) in enumerate(zip(methods,colors)):
            row=sub.loc[method];ax.plot([row.lower,row.upper],[j,j],'-o',lw=2,ms=4,color=col)
            ax.text(row.upper+.18,j,f'[{row.lower}, {row.upper}]',va='center',fontsize=8,color=col)
        ax.axvline(4,ls=':',lw=1,color='#555555');ax.set_xlim(-.3,11);ax.set_xticks(range(0,11,2));ax.set_title(policy,fontsize=10);ax.set_xlabel('Edge replacements');ax.grid(axis='x',color='#e4e8eb',lw=.6)
    axs[0].set_yticks(range(len(methods)),methods);axs[0].invert_yaxis();fig.tight_layout();save(fig,'wine_intervals')

if __name__=='__main__':
    overview();cut_cycle();wine()
    if (ROOT/'results'/'combined_summary.csv').exists():simulations()
