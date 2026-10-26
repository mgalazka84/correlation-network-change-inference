"""Reproducible explanatory figures for the two-period tree methodology.

Figures are mathematical illustrations, not simulated or empirical results.
The introductory witnesses are checked against every labelled four-node tree.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from network_bounds import complete_edges
from validate_bounds import all_trees

BLUE='#21618c'
ORANGE='#b85a1b'
GREY='#86939c'
DARK='#243746'


def save(fig,root,name):
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(root/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def tree(ax,edges,chosen,common,color):
    xy=np.array([[0.,1.],[1.,1.],[0.,0.],[1.,0.]])
    for i in sorted(chosen):
        u,v=edges[i];a,b=xy[u],xy[v]
        ax.plot([a[0],b[0]],[a[1],b[1]],color=GREY if i in common else color,
                lw=1.8 if i in common else 2.8,zorder=1)
    ax.scatter(xy[:,0],xy[:,1],s=215,color='white',edgecolors=DARK,lw=1.1,zorder=2)
    for label,point in zip('ABCD',xy):
        ax.text(*point,label,ha='center',va='center',fontsize=9,color=DARK,zorder=3)
    ax.set_xlim(-.22,1.22);ax.set_ylim(-.18,1.18);ax.set_aspect('equal');ax.axis('off')


def main(root):
    root=Path(root)/'figures';root.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    edges=complete_edges(4);trees=all_trees(4,edges)
    weights=[np.array([.79,.79,.8,.8,.8,.8]),np.array([.8,.8,.8,.8,.79,.79])]
    families=[]
    for w in weights:
        costs=[sum(w[list(t)]) for t in trees]
        families.append([t for t,c in zip(trees,costs) if np.isclose(c,min(costs),atol=1e-12,rtol=0)])
    # Witnesses: AB,AC,BD versus AB,BD,CD; AB,AC,AD versus AD,BD,CD.
    pairs=[({0,1,4},{0,4,5}),({0,1,2},{2,4,5})]
    attainable={len(t-u) for t in families[0] for u in families[1]}
    assert attainable=={1,2}
    for (t,u),h in zip(pairs,[1,2]):
        assert t in families[0] and u in families[1] and len(t-u)==h
    fig=plt.figure(figsize=(7.4,1.75))
    gs=fig.add_gridspec(1,4,left=.02,right=.98,bottom=.23,top=.82,wspace=.08)
    for group,((t,u),num) in enumerate(zip(pairs,[1,2])):
        title=['Closest pair: 1 replacement','Farthest pair: 2 replacements'][group]
        fig.text(.26+.48*group,.94,title,ha='center',color=DARK,fontsize=10,weight='bold')
        for col,chosen,color in [(0,t,ORANGE),(1,u,BLUE)]:
            ax=fig.add_subplot(gs[0,2*group+col]);tree(ax,edges,chosen,t&u,color)
            ax.text(.5,-.12,f'Period {col+1}',transform=ax.transAxes,ha='center',fontsize=9,color=DARK)
    fig.text(.5,.015,'Grey: shared edge    Colour: replaced edge',ha='center',fontsize=8,color=DARK)
    save(fig,root,'intro_turnover')

    fig,ax=plt.subplots(figsize=(7.4,2.65));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(.15,.2,9.7,3.55,'#edf3f8',BLUE),(.55,.6,7.2,2.35,'#dbe9ed','#527d89'),(.95,.98,5.35,1.15,'#ffffff',DARK)]
    for x,y,w,h,fc,ec in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.08',fc=fc,ec=ec,lw=1.2))
    ax.text(.4,3.35,r'Outer family: every spanning tree with $F \subseteq T \subseteq P$',fontsize=10,color=DARK)
    ax.text(.8,2.58,'Trees optimal for some weights in the confidence region',fontsize=10,color=DARK)
    ax.text(1.2,1.65,'All population-optimal trees',fontsize=10,weight='bold',color=DARK)
    ax.text(1.2,1.25,'Included when the input confidence event holds',fontsize=8.5,color=DARK)
    ax.annotate('Optimize the\nouter family\nexactly',xy=(9.2,.25),xytext=(8.8,1.65),
                ha='center',fontsize=9,color=BLUE,arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.1})
    fig.tight_layout(pad=.25);save(fig,root,'envelope_families')

    fig,ax=plt.subplots(figsize=(7.4,2.85))
    labels=['Every pair within tolerance','More change than tolerance','Unresolved']
    for y,lo,hi,c in [(2,0,1,BLUE),(1,3,5,ORANGE),(0,0,5,GREY)]:
        ax.plot([lo,hi],[y,y],color=c,lw=5,solid_capstyle='round')
        ax.scatter([lo,hi],[y,y],color=c,s=38,zorder=3)
        ax.text((lo+hi)/2,y+.18,f'[{lo}, {hi}]',ha='center',fontsize=10,color=DARK)
    ax.axvline(2,color=DARK,lw=1,ls='--');ax.text(2,2.6,'Tolerance k = 2',ha='center',fontsize=9,color=DARK,bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.set_yticks([2,1,0],labels);ax.set_xticks(range(8));ax.set_xlim(-.25,7.2);ax.set_ylim(-.5,2.9)
    ax.set_xlabel('Number of edge replacements');ax.spines[['top','left','right']].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=12);ax.grid(axis='x',alpha=.12)
    fig.tight_layout(pad=.5);save(fig,root,'turnover_decisions')
    print('Three explanatory figures generated; exact introductory witnesses verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
