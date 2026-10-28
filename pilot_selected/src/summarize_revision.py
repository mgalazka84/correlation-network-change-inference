"""Numerical summaries, paired contrasts, and completeness checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm,binomtest
from revision_methods import adjusted_holm

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_results'


def read(path):
    d=pd.read_csv(path,low_memory=False)
    for name in ['detect','coverage','envelope_coverage','input_error','fallback','change']:
        if name in d and d[name].dtype==object:
            d[name]=d[name].map({True:True,False:False,'True':True,'False':False})
            assert d[name].notna().all(),name
    return d


def summarize(d,keys):
    d=d.copy();d['false_change']=np.where((d.truth_lower==0)&(d.truth_upper==0),d.detect.astype(float),np.nan)
    rules=dict(reps=('rep','size'),coverage=('coverage','mean'),envelope_coverage=('envelope_coverage','mean'),
               input_error=('input_error','mean'),detect=('detect','mean'),false_change=('false_change','mean'),
               lower=('lower','mean'),upper=('upper','mean'),width=('width','mean'),seconds=('seconds','mean'))
    for col in ['M','pilot_correct','pilot_M','gap_min','gap_max','true_M','omitted','optimizer_seconds']:
        if col in d: rules[col]=(col,'mean')
    if 'fallback' in d: rules['fallback']=('fallback','sum')
    s=d.groupby(keys,sort=True).agg(**rules).reset_index();z=norm.ppf(.975)
    for name in ['coverage','envelope_coverage','input_error','detect','false_change']:
        v=s[name];n=s.reps;den=1+z*z/n;center=(v+z*z/(2*n))/den;radius=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
        s[name+'_lo']=np.maximum(0,center-radius);s[name+'_hi']=np.minimum(1,center+radius)
    return s


def contrasts(d,methods=('Five-splits','Local-E64')):
    a=d[d.method==methods[0]];b=d[d.method==methods[1]]
    z=a.merge(b,on=['cell','rep','seed'],suffixes=('_a','_b'),validate='one_to_one');out=[]
    for ci,g in z.groupby('cell'):
        # Only changed, unique-tree targets are a detection comparison.
        if 'truth_lower_a' in g and not g.truth_lower_a.iloc[0]>0:continue
        x=g.detect_a.to_numpy(int);y=g.detect_b.to_numpy(int);diff=x-y
        b10=int(((x==1)&(y==0)).sum());b01=int(((x==0)&(y==1)).sum())
        pvalue=binomtest(b10,b10+b01,.5).pvalue if b10+b01 else 1.
        se=diff.std(ddof=1)/len(g)**.5
        out.append(dict(cell=ci,n=len(g),method_a=methods[0],method_b=methods[1],
                        detect_a=x.mean(),detect_b=y.mean(),only_a=b10,only_b=b01,
                        difference=diff.mean(),paired_se=se,
                        descriptive_wald_lo=diff.mean()-norm.ppf(.975)*se,
                        descriptive_wald_hi=diff.mean()+norm.ppf(.975)*se,pvalue=pvalue))
    out=pd.DataFrame(out);out['holm_pvalue']=adjusted_holm(out.pvalue.to_numpy())
    return out


def integrity(d,name,cells,methods,reps=250):
    keys=['cell','rep','method']+(['gamma','K'] if 'gamma' in d else [])
    assert not d.duplicated(keys).any(),name
    assert ((0<=d.lower)&(d.lower<=d.upper)&(d.upper<=d.p-1)).all(),name
    assert len(d)==cells*methods*reps,(name,len(d))
    group=['cell','method']+(['gamma','K'] if 'gamma'in d else [])
    assert (d.groupby(group).size()==reps).all(),name
    assert d[['coverage','envelope_coverage','input_error','detect']].notna().all().all()
    return dict(rows=len(d),datasets=len(d[['cell','rep']].drop_duplicates()),
                cells=d.cell.nunique(),method_settings=methods,reps=reps,passed=True)


def main():
    expanded=read(R/'expanded_replications.csv');tuning=read(R/'tuning_replications.csv')
    boost=read(R/'original-bootstrap_replications.csv')
    old=pd.concat([read(ROOT/'results/simulation_replications.csv'),read(ROOT/'results/legacy_replications.csv'),boost],ignore_index=True)
    # The inherited-method archive omits target columns; recover the same
    # population targets from the original design, not from observed outputs.
    targets=old[['cell','truth_lower','truth_upper']].dropna().drop_duplicates()
    assert not targets.cell.duplicated().any(), 'Conflicting original targets'
    targets=targets.set_index('cell')
    for col in ['truth_lower','truth_upper']:
        old[col]=old[col].fillna(old.cell.map(targets[col]))
        assert old[col].notna().all(), 'Missing original target'
    e64=read(ROOT/'results/e64_replications.csv');e64['truth_lower']=e64.p-2;e64['truth_upper']=e64.p-2
    checks=dict(expanded=integrity(expanded,'expanded',38,8),
                tuning=integrity(tuning,'tuning',6,20),original_bootstrap=integrity(boost,'original bootstrap',38,1))
    assert set(tuning.seed)<=set(expanded.seed)
    assert not (set(expanded.seed)&set(old.seed)), 'New/old seed overlap'
    es=summarize(expanded,['cell','method']);es.to_csv(R/'expanded_summary.csv',index=False)
    ts=summarize(tuning,['cell','gamma','K','method']);ts.to_csv(R/'tuning_summary.csv',index=False)
    os=summarize(old,['cell','p','n','rho','model','change','method']);os.to_csv(R/'original_with_bootstrap_summary.csv',index=False)
    contrasts(pd.concat([old,e64],ignore_index=True)).to_csv(R/'paired_original_e64.csv',index=False)
    contrasts(expanded).to_csv(R/'paired_revision_e64.csv',index=False)
    contrasts(expanded,('Five-splits','Multiplier')).to_csv(R/'paired_revision_multiplier.csv',index=False)
    null=expanded[(expanded.truth_lower==0)&(expanded.truth_upper==0)]
    null.groupby(['cell','method']).agg(reps=('rep','size'),false_changes=('detect','sum')).reset_index().to_csv(R/'false_changes.csv',index=False)
    for filename in ['expanded_replications.csv','tuning_replications.csv','original-bootstrap_replications.csv']:
        checks[filename+'_sha256']=hashlib.sha256((R/filename).read_bytes()).hexdigest()
    (R/'revision_integrity.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
