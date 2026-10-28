"""Generate manuscript tables from complete replication records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';TEX=ROOT/'manuscript'
OUT.mkdir(parents=True,exist_ok=True);TEX.mkdir(parents=True,exist_ok=True)
METHODS=['Hoeffding','All-pairs','Local-McNemar','Local-E8','One-split','Five-splits']

def main():
    a=pd.read_csv(OUT/'simulation_replications.csv');b=pd.read_csv(OUT/'legacy_replications.csv');df=pd.concat([a,b],ignore_index=True)
    assert len(df)==57000 and not df.duplicated(['cell','rep','method']).any()
    assert df.groupby(['cell','method']).size().eq(250).all()
    assert df.cell.nunique()==38 and df.method.nunique()==6
    assert df[['lower','upper','width']].notna().all().all()
    assert (df.lower<=df.upper).all() and (df.width==df.upper-df.lower).all()
    assert (df.lower>=0).all() and (df.upper<=df.p-1).all()
    s=df.groupby(['cell','p','n','rho','model','change','method'],sort=False).agg(reps=('rep','size'),coverage=('coverage','mean'),envelope_coverage=('envelope_coverage','mean'),input_error=('input_error','mean'),detect=('detect','mean'),lower=('lower','mean'),upper=('upper','mean'),width=('width','mean'),seconds=('seconds','mean'),M=('M','mean'),fallback=('fallback','sum')).reset_index()
    z=norm.ppf(.975)
    for metric in ['coverage','envelope_coverage','input_error','detect']:
        rate=s[metric];N=s.reps;den=1+z*z/N;c=(rate+z*z/(2*N))/den;r=z*np.sqrt(rate*(1-rate)/N+z*z/(4*N*N))/den
        s[metric+'_lo']=np.maximum(0,c-r);s[metric+'_hi']=np.minimum(1,c+r)
    s.to_csv(OUT/'combined_summary.csv',index=False)
    strong=s[(s.p==8)&(s.n==1000)&(s.rho==.65)&(s.model=='Gaussian')&s.change].set_index('method')
    lines=[r'\begin{longtable}{lrrrrrrr}',r'\caption{All simulation cells. Err is input-error frequency, Env and Cov are envelope and turnover coverage, Det is the frequency of a positive lower bound, W is mean width, and Time is mean elapsed seconds. Each row contains 250 replications.}\label{tab:full}\\',r'\toprule Method & Err & Env & Cov & Det & W & Tests & Time\\\midrule\endfirsthead',r'\toprule Method & Err & Env & Cov & Det & W & Tests & Time\\\midrule\endhead',r'\bottomrule\endfoot']
    for ci,g in s.groupby('cell',sort=True):
        row=g.iloc[0];model=row['model'].replace('Student-t5',r'Student $t_5$');change='changed' if row['change'] else 'unchanged'
        lines.append(r'\multicolumn{8}{l}{\textbf{Cell '+str(ci)+r':} '+f'$p={row.p}$, $n={row.n}$, $\\rho={row.rho:.2f}$, {model}, {change}'+r'}\\*')
        g=g.set_index('method')
        for mi,m in enumerate(METHODS):
            q=g.loc[m];lines.append(m+' & '+' & '.join([f'{q.input_error:.3f}',f'{q.envelope_coverage:.3f}',f'{q.coverage:.3f}',f'{q.detect:.3f}',f'{q.width:.3f}',f'{q.M:.1f}',f'{q.seconds:.3f}'])+(r'\\*' if mi<5 else r'\\[3pt]'))
    lines.append(r'\end{longtable}')
    (TEX/'full_results.tex').write_text('\n'.join(lines))
    w=pd.read_csv(OUT/'wine_results.csv');rows=[r'\begin{table}[tb]\centering\caption{Wine-data turnover intervals. The estimated replacement count is four in both data versions. Sensitivity outputs do not establish independence after deduplication.}\label{tab:wine}',r'\begin{tabular}{lcc}\toprule Method & All records & Unique feature rows\\\midrule']
    for m in METHODS+(['Local-E64'] if 'Local-E64' in w.method.values else []):
        vals=[]
        for policy in ['All records','Unique feature rows']:
            q=w[(w.method==m)&(w.policy==policy)].iloc[0];vals.append(f'$[{q.lower},{q.upper}]$')
        rows.append(m+' & '+' & '.join(vals)+r'\\')
    rows.extend([r'\bottomrule\end{tabular}',r'\end{table}']);(TEX/'wine_table.tex').write_text('\n'.join(rows))
    sens=pd.read_csv(OUT/'wine_seed_sensitivity.csv');g=sens.groupby(['policy','method'])
    # Paired Monte Carlo contrasts are descriptive, with paired standard errors.
    contrasts=[]
    for ci,g in df.groupby('cell'):
        for metric in ['detect','width']:
            pv=g.pivot(index='rep',columns='method',values=metric).astype(float)
            for comparator in METHODS[:-2]:
                d=pv['Five-splits']-pv[comparator];contrasts.append(dict(cell=int(ci),metric=metric,comparator=comparator,difference=d.mean(),paired_standard_error=d.std(ddof=1)/np.sqrt(len(d))))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_monte_carlo_contrasts.csv',index=False)
    if (OUT/'e64_replications.csv').exists():
        e=pd.read_csv(OUT/'e64_replications.csv')
        es=e.groupby(['cell','p','model']).agg(detect=('detect','mean'),width=('width','mean'),coverage=('coverage','mean'),input_error=('input_error','mean')).reset_index()
        es.to_csv(OUT/'e64_summary.csv',index=False)
        erows=[r'\begin{table}[ht]\centering\caption{Additional Local-E64 benchmark. All cells have changed stars, $n=1000$, $\rho=0.65$, and 250 replications.}',r'\begin{tabular}{rlrrr}\toprule $p$ & Model & Detection & Mean width & Coverage\\\midrule']
        for _,q in es.iterrows():erows.append(f'{q.p} & {q.model} & {q.detect:.3f} & {q.width:.3f} & {q.coverage:.3f}'+r'\\')
        erows.extend([r'\bottomrule\end{tabular}\end{table}']);(TEX/'e64_results.tex').write_text('\n'.join(erows))
    report=dict(datasets=9500,method_outputs=len(df),all_turnover_covered=bool(df.coverage.all()),all_envelopes_covered=bool(df.envelope_coverage.all()),maximum_cell_input_error=float(s.input_error.max()),fallbacks=int(df.fallback.sum()),all_integrity_checks_passed=True)
    (OUT/'results_integrity.json').write_text(json.dumps(report,indent=2));print(report)
    print(s.groupby('method').agg(max_error=('input_error','max'),min_coverage=('coverage','min')).to_string())

if __name__=='__main__':main()
