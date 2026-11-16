"""Independent checks of saved external replay, sampling and selection."""
import argparse
from pathlib import Path
import math
import time
import numpy as np
from game2_lab.cross_arena import read,load_catalog,draw_population,sample_actions,native_module
from game2_lab.core import focal_samples
from game2_lab.human import payoff_arrays
from run_population_study import ROOT,save,sha
from verify_population_study import rational


def independent_cdf_gap(values,weights,public):
    order=np.argsort(values);v=values[order];w=np.cumsum(weights[order]/weights.sum())
    public=np.sort(public);knots=np.unique(np.r_[v,public]);answer=0.
    for side in ['left','right']:
        ix=np.searchsorted(v,knots,side=side)
        model=np.r_[0.,w][ix]
        observed=np.searchsorted(public,knots,side=side)/len(public)
        answer=max(answer,float(np.max(abs(model-observed))))
    return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/game2-cross-arena')
    args=parser.parse_args();out=args.study.resolve();start=time.perf_counter()
    freeze=read(out/'freeze.json');result=read(out/'results.json');cfg=freeze['configuration']
    assert sha(out/'freeze.json')==result['freeze_sha256']
    for path,digest in freeze['input_and_source_sha256'].items():assert sha(ROOT/path)==digest,path
    vendor=ROOT/'data/parry-replay';fit,catalog,fields=load_catalog(vendor)
    native=native_module(vendor)
    pool=np.array([r['allocation']for r in freeze['pool']],dtype=np.uint8)
    n=cfg['groups_per_field'];columns=[];foundation_cells=0;regenerated=0
    # Include random and earliest rows, so rare zero/tie patterns are not excluded on purpose.
    for j,field in enumerate(fields):
        with np.load(out/'fresh-fields'/(field['name']+'.npz'))as data:
            rows=data['triplets'];sums=data['rank_sum4'];seed=int(data['seed'])
        assert seed==cfg['fresh_candidate_seed']+j
        expected=draw_population(catalog,field['weights'],3*n,seed).reshape(n,3,5)
        np.testing.assert_array_equal(rows,expected);regenerated+=n
        assert np.all(rows.sum(axis=2)==100)
        ix=np.unique(np.r_[np.arange(128),np.random.default_rng(7151+j).choice(n,2048,replace=False)])
        quick,_=payoff_arrays(pool.tolist(),rows[ix])
        g=native.compress_groups(rows[ix]);pay=native.payoff_matrix(pool,g)
        counts=np.rint(g[2]*len(ix)).astype(np.uint64)
        native_sums=np.einsum('ij,j->i',pay,counts,dtype=np.uint64)//3
        np.testing.assert_array_equal(native_sums,quick.sum(axis=1,dtype=np.uint64))
        foundation_cells+=len(pool)*len(ix)
        columns.append(sums/(4*n))
        print('Verified field',field['name'],flush=True)
    v=np.column_stack(columns)
    with np.load(out/'fresh-scores.npz')as data:np.testing.assert_array_equal(v,data['values'])
    minima=v.min(axis=1)
    selected=min(range(len(pool)),key=lambda i:(-minima[i],tuple(pool[i])))
    assert pool[selected].tolist()==result['primary_catalog_leader']
    strict=np.array([f['strict']for f in fields])
    for label,mask in [('all_17',np.ones(len(fields),dtype=bool)),('strict_9',strict)]:
        sub=v[:,mask];floor=sub.min(axis=1);risk=(sub.max(axis=0)-sub).max(axis=1)
        a=min(range(len(pool)),key=lambda i:(-floor[i],tuple(pool[i])))
        b=min(range(len(pool)),key=lambda i:(risk[i],tuple(pool[i])))
        assert a==result['ranking'][label]['maximin_index']
        assert b==result['ranking'][label]['minimax_regret_index']
        np.testing.assert_array_equal(floor,result['ranking'][label]['minimum'])
        np.testing.assert_array_equal(risk,result['ranking'][label]['maximum_regret'])
    eps=9*math.sqrt(math.log(2*len(pool)*17/.05)/(2*n))
    assert eps==result['conditional_simultaneous_95_percent_cell_error_bound']
    # Independent full-group primary comparison on each fresh field.
    checked_full=0
    for j,f in enumerate(fields):
        with np.load(out/'fresh-fields'/(f['name']+'.npz'))as data:rows=data['triplets']
        for i in sorted({selected,freeze['named_indices']['our_v1'],freeze['named_indices']['parry_36_64']}):
            assert int(focal_samples(pool[i].tolist(),rows)['payout4'].sum())==round(v[i,j]*4*n)
            checked_full+=n
    original=read(out/'original-replay.json')
    differences=[abs(r['difference_from_original_audit'])for f in original['fields']for r in f['results']if 'difference_from_original_audit'in r]
    assert len(differences)==68 and max(differences)==0
    # All original candidates on every old bank, plus every new candidate on a foundation subset.
    old=read(ROOT/'results/game2-final/results.json');index={tuple(x):i for i,x in enumerate(pool.tolist())}
    old_ids=[index[tuple(r['allocation'])]for r in old['candidates'].values()]
    old_values=[];old_subset=0
    for j,key in enumerate(old['cell_keys']):
        with np.load(ROOT/'results/game2-final/evaluation'/(key+'.npz'))as data:rows=data['triplets'];expected=data['rank_sum4']
        with np.load(out/'old-stress'/(key+'.npz'))as data:sums=data['rank_sum4'];size=int(data['size'])
        np.testing.assert_array_equal(sums[old_ids],expected)
        ix=np.random.default_rng(1820+j).choice(len(rows),256,replace=False)
        direct,_=payoff_arrays(pool.tolist(),rows[ix]);g=native.compress_groups(rows[ix]);p=native.payoff_matrix(pool,g)
        counts=np.rint(g[2]*len(ix)).astype(np.uint64)
        np.testing.assert_array_equal(direct.sum(axis=1),np.einsum('ij,j->i',p,counts,dtype=np.uint64)//3)
        old_subset+=len(pool)*len(ix);old_values.append(sums/(4*size))
    old_values=np.column_stack(old_values)
    stress=read(out/'old-stress-summary.json')
    for name,w in old['policy']['block_weights'].items():
        weights=np.array([float(rational(w[k])) if k in w else 0.for k in old['cell_keys']])
        np.testing.assert_allclose(old_values@weights,stress['block_values'][name],rtol=0,atol=1e-12)
    # Reconstruct saved score distributions and independently check all CDF knots.
    cdf=read(out/'cdf-comparison.json');public=np.array(read(vendor/'public-scores.json')['scores'])
    cdf_cells=0;original_cdf_error=0.
    for phase in ['original','fresh']:
        per_family=cfg['cdf_'+phase+'_actions_per_family']
        own_seed=cfg['cdf_'+phase+'_action_seed']
        own,families=sample_actions(catalog,per_family,own_seed)
        for j,field in enumerate(fields):
            with np.load(out/('cdf-'+phase)/(field['name']+'.npz'))as data:
                u=data['values'];rows=data['opponent_triplets']
                np.testing.assert_array_equal(data['own_actions'],own)
                np.testing.assert_array_equal(data['own_families'],families)
                seed=int(data['opponent_seed'])
            # The complete own-score vector is original C++ output. Check stratified actions via our scorer.
            own_ix=np.unique(np.r_[np.arange(0,len(own),per_family),np.random.default_rng(811+j).choice(len(own),16,replace=False)])
            # Check their complete mean, not only a random subset of opponent groups.
            q,_=payoff_arrays(own[own_ix].tolist(),rows)
            np.testing.assert_allclose(q.mean(axis=1)/4,u[own_ix],rtol=0,atol=1e-10)
            cdf_cells+=len(own_ix)*len(rows)
            weights=np.array(field['weights'])[families]/per_family
            full=independent_cdf_gap(u,weights,public)
            r=cdf[phase][field['name']]
            assert abs(full-r['maximum_cdf_gap'])<1e-11
            coarse=np.array([weights[u<=t].sum()for t in fit['thresholds']])
            gap=abs(coarse-fit['observed_cdf'])
            assert abs(gap.max()-r['original_26_threshold_max_gap'])<1e-12
            assert abs(gap.mean()-r['original_26_threshold_mean_gap'])<1e-12
            assert abs(weights[u>public.max()+.2].sum()-r['mass_above_public_max_plus_point_two'])<1e-12
            if phase=='original':
                original_cdf_error=max(original_cdf_error,float(abs(coarse-field['original_model']['validation_cdf']).max()))
            print('Verified CDF',phase,field['name'],flush=True)
    assert original_cdf_error<1e-10
    for path,digest in freeze['input_and_source_sha256'].items():assert sha(ROOT/path)==digest,path
    save(out/'acceptance.json',{'passed':True,'freeze_sha256':sha(out/'freeze.json'),'results_sha256':sha(out/'results.json'),
        'source_sha256':sha(Path(__file__)),'regenerated_fresh_triplets':regenerated,
        'fresh_all_candidate_subset_cells_checked':foundation_cells,'fresh_full_primary_cells_checked':checked_full,
        'runner_full_named_cells_checked':result['independent_candidate_group_cells'],
        'old_all_candidate_subset_cells_checked':old_subset,'old_original_candidate_sums_checked':191*85,
        'cdf_full_opponent_cells_checked':cdf_cells,'maximum_original_cdf_reproduction_error':original_cdf_error,
        'original_replay_comparisons_matching_exactly':68,'elapsed_seconds':time.perf_counter()-start,
        'scope':'Numerical reproduction and scoring/selection verification only; no hidden-population identification or global full-action certificate.'})
    print('Accepted cross-arena study',flush=True)


if __name__=='__main__':main()
