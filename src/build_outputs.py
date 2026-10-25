"""Generate figures, tables and an analytic example from saved results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from network_bounds import complete_edges, compare_intervals, turnover_lower_bound, components

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MAN = ROOT / "manuscript"
FIG.mkdir(exist_ok=True)
MAN.mkdir(exist_ok=True)
(ROOT/'validation').mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "savefig.bbox":"tight"})


def analytic_example():
    p=4
    edges=complete_edges(p)
    w1=np.array([.79,.79,.80,.80,.80,.80])
    w2=np.array([.80,.80,.80,.80,.79,.79])
    fit=compare_intervals(p,edges,w1,w1,w2,w2)
    e1,e2=fit["envelope1"],fit["envelope2"]
    naive=max(len(e1[0]-e2[1]),len(e2[0]-e1[1]),components(p,edges,e1[1]&e2[1])-1)
    cert=turnover_lower_bound(p,edges,e1,e2,certificate=True)
    eigen=[]
    for w in [w1,w2]:
        r=np.eye(p)
        for (a,b),v in zip(edges,np.sin(np.pi*(1-w)/2)):
            r[a,b]=r[b,a]=v
        eigen.append(float(np.linalg.eigvalsh(r).min()))
    assert min(eigen)>0 and cert["turnover_lower"]==1 and naive==0
    data={"edges":edges,"weights1":w1.tolist(),"weights2":w2.tolist(),
          "forced1":sorted(e1[0]),"possible1":sorted(e1[1]),
          "forced2":sorted(e2[0]),"possible2":sorted(e2[1]),
          "elementary_lower_bound":naive,"certificate":cert,
          "degree_change_lower":fit["degree_change_lower"].tolist(),
          "degree_change_upper":fit["degree_change_upper"].tolist(),
          "raw_degree_change_lower":fit["raw_degree_change_lower"].tolist(),
          "raw_degree_change_upper":fit["raw_degree_change_upper"].tolist(),
          "latent_gaussian_min_eigenvalues":eigen}
    (ROOT/"validation"/"analytic_example.json").write_text(json.dumps(data,indent=2))
    loc=np.array([[0,1],[-.8,0],[.8,0],[0,-1]])
    fig,axes=plt.subplots(1,2,figsize=(7,3.2))
    for ax,(f,poss),title in zip(axes,[e1,e2],["Period 1: A has at least 2 links","Period 2: D has at least 2 links"]):
        for i in sorted(poss):
            a,b=edges[i]
            ax.plot(loc[[a,b],0],loc[[a,b],1],color="#176b85" if i in f else "#aebbc1",
                    lw=3 if i in f else 1.8,ls="-" if i in f else "--",zorder=1)
        ax.scatter(loc[:,0],loc[:,1],s=440,color="#f2f5f7",edgecolors="#193b4b",zorder=3)
        for i,(x,y) in enumerate(loc): ax.text(x,y,chr(65+i),ha="center",va="center",zorder=4,fontweight="bold")
        ax.set_title(title,fontsize=10)
        ax.set_xlim(-1.2,1.2);ax.set_ylim(-1.3,1.3);ax.set_aspect("equal");ax.axis("off")
    fig.text(.5,.02,"Solid: mandatory edge     Dashed: possible edge     Omitted B-C edge: impossible",ha="center",fontsize=8.5)
    fig.savefig(FIG/"analytic_example.pdf");fig.savefig(FIG/"analytic_example.png",dpi=160);plt.close(fig)


def pilot_figures_and_tables():
    s=pd.read_csv(ROOT/"results"/"pilot"/"summary.csv")
    cases=["clear_null","clear_change","near_null","near_change","exact_tie_null"]
    labels=["Clear\nnull","Clear\nchange","Near-tie\nnull","Near-tie\nchange","Exact-tie\nnull"]
    methods=["dependent_multiplier","bonferroni_hac","iid_multiplier"]
    colors=["#176b85","#e29338","#b54751"]
    names=["Dependent multiplier","Bonferroni + HAC","IID multiplier"]
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5),sharey=True)
    for ax,n in zip(axes,[250,1000]):
        for shift,method,color,label in zip([-.18,0,.18],methods,colors,names):
            d=s[(s.n==n)&(s.method==method)].set_index("case").loc[cases]
            y=d.box_covered.to_numpy()
            error=np.vstack([y-d.box_covered_wilson_low.to_numpy(),d.box_covered_wilson_high.to_numpy()-y])
            ax.errorbar(np.arange(5)+shift,y,yerr=error,fmt="o",ms=4,capsize=2,color=color,label=label)
        ax.axhline(.95,color="#243c48",ls="--",lw=1)
        ax.set_xticks(range(5),labels,fontsize=8)
        ax.set_title(f"n = {n} per period")
        ax.set_ylim(.2,1.01)
        ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Simultaneous input coverage")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles,labs,loc="lower center",ncol=3,bbox_to_anchor=(.5,-.08),frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"input_coverage.pdf");fig.savefig(FIG/"input_coverage.png",dpi=160);plt.close(fig)
    # Directly show that abstention near ties is accompanied by loss of power.
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.4))
    for ax,n in zip(axes,[250,1000]):
        d=s[(s.n==n)&(s.method=="dependent_multiplier")].set_index("case").loc[cases]
        xx=np.arange(5)
        ax.bar(xx-.18,d.plugin_changed,.35,label="Plug-in trees differ",color="#b6c3c9")
        ax.bar(xx+.18,d.detected,.35,label="Positive structural lower bound",color="#176b85")
        ax.set_xticks(xx,labels,fontsize=8);ax.set_ylim(0,1.08)
        ax.set_title(f"n = {n} per period");ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Proportion of 200 repetitions")
    handles,labs=axes[0].get_legend_handles_labels()
    fig.legend(handles,labs,loc="lower center",ncol=2,bbox_to_anchor=(.5,-.06),frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"detection.pdf");fig.savefig(FIG/"detection.png",dpi=160);plt.close(fig)
    labels_tex={"clear_null":"Clear null","clear_change":"Clear change","near_null":"Near-tie null", "near_change":"Near-tie change","exact_tie_null":"Exact-tie null"}
    lines=[r"\begin{tabular}{llrrrrr}",r"\toprule",
           r"Scenario & $n$ & Input cov. & Detect. & Mean $L_H$ & Width & Plug-in diff. \\",r"\midrule"]
    for n in [250,1000]:
        for case in cases:
            a=s[(s.n==n)&(s.method=="dependent_multiplier")&(s.case==case)].iloc[0]
            lines.append(f"{labels_tex[case]} & {n} & {100*a.box_covered:.1f}\\% & {100*a.detected:.1f}\\% & {a.turnover_lower:.3f} & {a.degree_width:.2f} & {100*a.plugin_changed:.1f}\\% " + chr(92)*2)
        if n==250: lines.append(r"\midrule")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (MAN/"pilot_table.tex").write_text("\n".join(lines))
    print(json.dumps({"analytic_example":"created","figures":3,"table_rows":10}))


if __name__=="__main__":
    analytic_example()
    pilot_figures_and_tables()
