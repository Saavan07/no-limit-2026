"""Equal precision for the complete, already-frozen C3/C4 family."""
from pathlib import Path
import numpy as np
from game2_lab.cross_arena import read,native_module,counts_from_groups
from game2_lab.catalog_stress import paired_moments,empirical_bernstein_error
from run_population_study import ROOT,save,sha,write_csv


def main():
    out=ROOT/'results/game2-catalog-stress';dest=out/'refinement';dest.mkdir(exist_ok=True)
    freeze=read(out/'freeze.json');cfg=freeze['configuration'];models=[r for r in read(out/'model-validation.json')['models']if r['validation']['passes']]
    xs=np.array([[0,0,b,100-b,0]for b in range(101)],np.uint8)
    pool=np.array([r['allocation']for r in freeze['pool']],np.uint8)
    for x in xs.tolist():assert x in pool.tolist(),'No new candidate admissions'
    definition={'candidate_freeze_sha256':sha(out/'freeze.json'),'model_validation_sha256':sha(out/'model-validation.json'),
        'allocations':xs.tolist(),'sample_source':'Same saved million-group banks; all 101 C3/C4 actions were in the initial 760-vector freeze.',
        'alpha_per_class':.05/3,'confidence_classes':['All 760 prefix means','All 101 C2/C4 full means and unordered pairs','All 760 potential full-group means and unordered pairs, covering the data-selected C3/C4 subset'],
        'source_sha256':sha(Path(__file__)),'scope':'Numerical precision refinement of frozen actions, not new population research or a fresh holdout.'}
    path=dest/'protocol.json'
    if path.exists():assert read(path)==definition
    else:save(path,definition)
    native=native_module(ROOT/'data/parry-replay');values=[];moments=[];n=cfg['split_groups'];alpha=.05/3
    for j,row in enumerate(models):
        path=dest/(row['name']+'.npz')
        if not path.exists():
            with np.load(out/'evaluation'/(row['name']+'.npz'))as d:rows=d['triplets'];oldsum=d['split_rank_sum4'][0];oldcross=d['split_cross_sum16'][0,0]
            g=native.compress_groups(rows);q=native.payoff_matrix(xs,g);sums,cross=paired_moments(q,counts_from_groups(g,n))
            assert sums[0]==oldsum and cross[0,0]==oldcross
            np.savez_compressed(path,rank_sum4=sums,cross_sum16=cross,size=n)
        with np.load(path)as d:values.append(d['rank_sum4']/(4*n));moments.append(d['cross_sum16'])
        print('REFINE C3/C4',j+1,'/',len(models),flush=True)
    v=np.array(values).T;moments=np.array(moments);comparisons=len(models)*(len(pool)+len(pool)*(len(pool)-1)//2)
    variance=(moments[:,np.arange(101),np.arange(101)]/16-n*v.T**2)/(n-1)
    errors=empirical_bernstein_error(variance,n,comparisons,alpha,9)
    lower=(v.T-errors).min(axis=0);upper=(v.T+errors).min(axis=0);b=int(v.min(axis=1).argmax())
    with np.load(out/'candidate-scores.npz')as d:cv=d['values']
    with np.load(out/'split-scores.npz')as d:v24=d['values'];mom24=d['cross_sum16']
    e24=empirical_bernstein_error((mom24[:,np.arange(101),np.arange(101)]/16-n*v24.T**2)/(n-1),n,len(models)*(101+5050),alpha,9)
    low24=(v24.T-e24).min(axis=0);up24=(v24.T+e24).min(axis=0)
    eps=9*np.sqrt(np.log(2*len(pool)*len(models)/alpha)/(2*cfg['candidate_groups']))
    outside=np.array([not(x[0]==x[1]==x[4]==0)for x in pool.tolist()])
    best_outside_upper=float((cv[outside].min(axis=1)+eps).max())
    pair_errors=[]
    for j,matrix in enumerate(moments):
        square=(np.diag(matrix)[:,None]+np.diag(matrix)[None,:]-2*matrix.astype(float))/16
        var=np.maximum((square-n*(v[:,j,None]-v[None,:,j])**2)/(n-1),0)
        pair_errors.append(empirical_bernstein_error(var,n,comparisons,alpha,18))
    np.savez_compressed(dest/'scores.npz',values=v,allocations=xs,names=[r['name']for r in models],cross_sum16=moments,
        mean_error=errors,pair_error=pair_errors,c24_mean_error=e24)
    result={'allocation':xs[b].tolist(),'C3':b,'C4':100-b,'minimum':float(v[b].min()),'maximum':float(v[b].max()),
        'weakest_field':models[int(v[b].argmin())]['name'],'sampling_plausible_C3_values':np.flatnonzero(upper>=lower.max()).tolist(),
        'c34_robust_lower':lower.tolist(),'c34_robust_upper':upper.tolist(),'c24_robust_lower':low24.tolist(),'c24_robust_upper':up24.tolist(),
        'sampling_plausible_C2_values_joint_with_c34':np.flatnonzero(up24>=low24.max()).tolist(),
        'best_c34_robust_lower':float(lower.max()),'best_c24_robust_upper':float(up24.max()),
        'best_non_c34_prefix_upper':best_outside_upper,'all_760_prefix_cell_bound':float(eps),
        'certified_over_non_c34_frozen_actions':bool(lower.max()>best_outside_upper),
        'full_group_comparison_universe':len(pool),'joint_confidence':.95,'confidence_scope':'Union over the three declared numerical classes, conditional on the 87 retained laws and frozen candidate actions. Excludes model error and all omitted actions.',
        'scope':definition['scope']}
    save(dest/'results.json',result)
    write_csv(dest/'all-101-C3-C4-splits.csv',['C3','C4','minimum','maximum','lower','upper','worst_field'],
        [[bid,100-bid,float(v[bid].min()),float(v[bid].max()),float(lower[bid]),float(upper[bid]),models[int(v[bid].argmin())]['name']]for bid in range(101)])
    print('REFINED',result,flush=True)


if __name__=='__main__':main()
