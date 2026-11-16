"""Independent audit of all C3/C4 refinements and joint numerical bounds."""
from pathlib import Path
import time
import numpy as np
from game2_lab.cross_arena import read,native_module,uncompressed_groups
from game2_lab.human import payoff_arrays
from run_population_study import ROOT,save,sha


def main():
    out=ROOT/'results/game2-catalog-stress';dest=out/'refinement';result=read(dest/'results.json');protocol=read(dest/'protocol.json');start=time.perf_counter()
    freeze=read(out/'freeze.json');assert protocol['candidate_freeze_sha256']==sha(out/'freeze.json')
    assert protocol['model_validation_sha256']==sha(out/'model-validation.json')
    assert protocol['source_sha256']==sha(ROOT/'run_game2_catalog_refinement.py')
    models=[r for r in read(out/'model-validation.json')['models']if r['validation']['passes']]
    xs=protocol['allocations'];pool=[r['allocation']for r in freeze['pool']]
    assert xs==[[0,0,b,100-b,0]for b in range(101)]and all(x in pool for x in xs)
    native=native_module(ROOT/'data/parry-replay');values=[];matrices=[];cells=0;full_cells=0
    check=sorted({35,36,37,38,result['C3']})
    for j,row in enumerate(models):
        with np.load(out/'evaluation'/(row['name']+'.npz'))as d:rows=d['triplets'];n=int(d['size'])
        with np.load(dest/(row['name']+'.npz'))as d:sums=d['rank_sum4'];matrix=d['cross_sum16']
        full,_=payoff_arrays([xs[b]for b in check],rows)
        np.testing.assert_array_equal(full.sum(axis=1),sums[check])
        q=full.astype(float);np.testing.assert_array_equal(q@q.T,matrix[np.ix_(check,check)])
        full_cells+=len(check)*n
        ix=np.random.default_rng(1287+j).choice(n,1024,replace=False)
        direct,_=payoff_arrays(xs,rows[ix]);quick=native.payoff_matrix(xs,uncompressed_groups(rows[ix]))//3
        np.testing.assert_array_equal(quick,direct);cells+=101*len(ix)
        values.append(sums/(4*n));matrices.append(matrix)
        print('Verified refinement',j+1,'/',len(models),flush=True)
    v=np.array(values).T;matrices=np.array(matrices);alpha=.05/3;log=np.log(4*len(models)*(760+760*759//2)/alpha)
    var=np.maximum((matrices[:,np.arange(101),np.arange(101)]/16-n*v.T**2)/(n-1),0)
    errors=np.sqrt(2*var*log/n)+7*9*log/(3*(n-1));lower=(v.T-errors).min(axis=0);upper=(v.T+errors).min(axis=0)
    b=int(v.min(axis=1).argmax());assert result['allocation']==xs[b]
    np.testing.assert_allclose(result['c34_robust_lower'],lower,rtol=0,atol=1e-12)
    np.testing.assert_allclose(result['c34_robust_upper'],upper,rtol=0,atol=1e-12)
    assert np.flatnonzero(upper>=lower.max()).tolist()==result['sampling_plausible_C3_values']
    expected_pair=[]
    for j,matrix in enumerate(matrices):
        square=(np.diag(matrix)[:,None]+np.diag(matrix)[None,:]-2*matrix.astype(float))/16
        var=np.maximum((square-n*(v[:,j,None]-v[None,:,j])**2)/(n-1),0)
        expected_pair.append(np.sqrt(2*var*log/n)+7*18*log/(3*(n-1)))
    with np.load(out/'split-scores.npz')as d:v24=d['values'];mom24=d['cross_sum16']
    var24=np.maximum((mom24[:,np.arange(101),np.arange(101)]/16-n*v24.T**2)/(n-1),0)
    log24=np.log(4*len(models)*(101+5050)/alpha)
    error24=np.sqrt(2*var24*log24/n)+7*9*log24/(3*(n-1))
    np.testing.assert_allclose((v24.T-error24).min(axis=0),result['c24_robust_lower'],rtol=0,atol=1e-12)
    np.testing.assert_allclose((v24.T+error24).min(axis=0),result['c24_robust_upper'],rtol=0,atol=1e-12)
    with np.load(dest/'scores.npz')as d:
        np.testing.assert_array_equal(v,d['values'])
        np.testing.assert_allclose(errors,d['mean_error'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(expected_pair,d['pair_error'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(error24,d['c24_mean_error'],rtol=0,atol=1e-12)
    with np.load(out/'candidate-scores.npz')as d:cv=d['values']
    eps=9*np.sqrt(np.log(2*760*len(models)/alpha)/(2*freeze['configuration']['candidate_groups']))
    outside=np.array([not(x[0]==x[1]==x[4]==0)for x in pool]);maximum=float((cv[outside].min(axis=1)+eps).max())
    assert abs(maximum-result['best_non_c34_prefix_upper'])<1e-12
    assert result['certified_over_non_c34_frozen_actions']==bool(lower.max()>maximum)
    save(dest/'acceptance.json',{'passed':True,'results_sha256':sha(dest/'results.json'),'protocol_sha256':sha(dest/'protocol.json'),
        'source_sha256':sha(Path(__file__)),'full_C3_C4_foundation_cells':full_cells,'all_C3_C4_subset_cells':cells,
        'elapsed_seconds':time.perf_counter()-start,'scope':result['confidence_scope']})
    print('Accepted refinement',flush=True)


if __name__=='__main__':main()
