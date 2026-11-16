"""Post-selection model-removal diagnostics using the saved full-action scores."""
import numpy as np
from game2_lab.cross_arena import read
from game2_lab.fullscan import allocations,allocation_indices
from run_population_study import ROOT,save,write_csv


def main():
    out=ROOT/'results/game2-fullspace-search';f=read(out/'freeze.json');models=f['models'];xs=allocations();size=len(xs)
    first=np.full(size,np.inf);second=first.copy();owner=np.full(size,-1,dtype=np.int16);bycatalog={}
    for j,m in enumerate(models):
        if not m['qualified']:continue
        with np.load(out/'selection'/(m['name']+'.npz'))as d:u=d['rank_sum4'].astype(float)/(4*int(d['size']))
        smaller=u<first;second=np.where(smaller,first,np.minimum(second,u));first=np.minimum(first,u);owner[smaller]=j
        if m['catalog_key']not in bycatalog:bycatalog[m['catalog_key']]=u.copy()
        else:np.minimum(bycatalog[m['catalog_key']],u,out=bycatalog[m['catalog_key']])
    with np.load(out/'selection-objectives.npz')as d:np.testing.assert_array_equal(first,d['minimum__qualified'])
    baselines=[[0,2,35,63,0],[0,0,36,64,0],[0,36,0,64,0]];ids=allocation_indices(baselines);rows=[]
    def record(label,u,remaining):
        i=int(u.argmax());rows.append({'removal':label,'remaining_models':remaining,'allocation':xs[i].tolist(),'minimum':float(u[i]),
            'selected_floor':float(u[ids[0]]),'previous_C3_C4_floor':float(u[ids[1]]),'previous_C2_C4_floor':float(u[ids[2]])})
    count=sum(m['qualified']for m in models)
    for j,m in enumerate(models):
        if m['qualified']:record('field:'+m['name'],np.where(owner==j,second,first),count-1)
    for key in bycatalog:
        u=np.full(size,np.inf)
        for name,values in bycatalog.items():
            if name!=key:np.minimum(u,values,out=u)
        record('catalog:'+key,u,sum(m['qualified']and m['catalog_key']!=key for m in models))
    for label,excluded in [('both_targeted_variants',{'c24_threshold_family','c4_threshold_upweight'}),
        ('all_equal_centered',{m['catalog_key']for m in models if m['equal_centered']})]:
        u=np.full(size,np.inf)
        for key,values in bycatalog.items():
            if key not in excluded:np.minimum(u,values,out=u)
        record(label,u,sum(m['qualified']and m['catalog_key']not in excluded for m in models))
    save(out/'model-removal-diagnostics.json',{'scope':'Post-selection sensitivity on known complete-action selection scores. No new holdout, candidate admission, or recommendation change.',
        'rows':rows,'single_model_omissions_preserving_exact_choice':sum(r['allocation']==baselines[0]for r in rows[:count]),
        'single_model_omissions':count})
    write_csv(out/'model-removal-diagnostics.csv',list(rows[0]),[[r[k]for k in rows[0]]for r in rows])
    print('Saved',len(rows),'complete-space removal diagnostics',flush=True)


if __name__=='__main__':main()
