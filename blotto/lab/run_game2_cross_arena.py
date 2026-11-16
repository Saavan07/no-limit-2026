"""Replay the supplied numerical models and compare the frozen joint pool."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import platform
import shutil
import time
import numpy as np
from game2_lab.cross_arena import (read, load_catalog, draw_population, sample_actions,
    native_module, exact_sample_sums, summary, fixed_pool, uncompressed_groups)
from game2_lab.handoff import cdf_diagnostic
from game2_lab.core import focal_samples
from run_population_study import ROOT, save, sha, write_csv
from verify_population_study import rational


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/game2-cross-arena')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = read(ROOT/'config/game2-cross-arena.json')
    vendor = ROOT/'data/parry-replay'
    fit, catalog, fields = load_catalog(vendor)
    assert len(fields) == 17 and sum(r['strict'] for r in fields) == 9 and len(catalog) == 40
    native = native_module(vendor)
    source_files = [Path(__file__), ROOT/'game2_lab/cross_arena.py', ROOT/'config/game2-cross-arena.json']
    input_files = [ROOT/'results/game2-handoff-integration/pool-freeze.json', ROOT/'results/game2-final/candidate-freeze.json', ROOT/'results/game2-selected-v1.json', ROOT/'results/game2-final/results.json']
    input_files += [vendor / p for p in read(vendor/'provenance.json')['files']]
    input_files += [vendor/'provenance.json', vendor/'public-scores.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files+input_files}
    pool = fixed_pool(read(input_files[0]), read(vendor/'results/final_audit.json'), read(vendor/'results/memory_equilibrium_best.json'))
    actions = np.asarray([r['allocation'] for r in pool], dtype=np.uint8)
    named = {'our_v1': [15,18,49,12,6], 'our_lowest_model': [20,24,44,5,7],
             'parry_36_64': [0,36,0,64,0], 'parry_39_61': [0,0,39,61,0],
             'parry_scout': [0,3,35,57,5], 'all_in_five': [0,0,0,0,100],
             'parry_earlier': [15,4,6,6,69], 'checkpoint_response': [14,18,27,16,25]}
    ids = {name: next(i for i, row in enumerate(pool) if row['allocation'] == x) for name,x in named.items()}
    freeze = {'created_utc': datetime.now(timezone.utc).isoformat(), 'configuration': config,
              'pool': pool, 'named_indices': ids, 'field_names': [f['name'] for f in fields],
              'input_and_source_sha256': hashes, 'versions': {'python': platform.python_version(), 'numpy': np.__version__},
              'scope': config['scope']}
    freeze_path = out/'freeze.json'
    if freeze_path.exists():
        old = read(freeze_path)
        for k in ['configuration','pool','named_indices','field_names','input_and_source_sha256','versions']:
            if old[k] != freeze[k]:
                raise ValueError('Cannot resume changed study: '+k)
    else:
        save(freeze_path, freeze)
        for p in source_files:
            dest = out/'source-snapshot'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    print('Frozen',len(pool),'candidates before fresh draws; SHA',sha(freeze_path),flush=True)
    start = time.perf_counter()
    n = config['groups_per_field']
    scores = []
    for j, field in enumerate(fields):
        path = out/'fresh-fields'/(field['name']+'.npz')
        if not path.exists():
            rows = draw_population(catalog, field['weights'], 3*n, config['fresh_candidate_seed']+j).reshape(n,3,5)
            groups = native.compress_groups(rows)
            sums = exact_sample_sums(native, actions, groups, n)
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, triplets=rows, rank_sum4=sums, size=n, seed=config['fresh_candidate_seed']+j)
        with np.load(path) as data:
            scores.append(data['rank_sum4'].copy()/(4*n))
        print('Fresh field',j+1,field['name'],'seconds',round(time.perf_counter()-start,1),flush=True)
    scores = np.column_stack(scores)
    ranking = summary(scores,fields,actions)
    selected = ranking['all_17']['maximin_index']
    ids['catalog_maximin'] = selected
    ids['catalog_minimax_regret'] = ranking['all_17']['minimax_regret_index']
    ids['strict_maximin'] = ranking['strict_9']['maximin_index']
    np.savez_compressed(out/'fresh-scores.npz', allocations=actions, values=scores, names=[f['name'] for f in fields])
    # Exact cross-engine comparison on every actual group for named entries and selected rules.
    detail, checks = {}, 0
    for j,field in enumerate(fields):
        with np.load(out/'fresh-fields'/(field['name']+'.npz')) as data:
            rows=data['triplets']
        groups=uncompressed_groups(rows)
        record={}
        for i in sorted(set(ids.values())):
            x=actions[i].tolist()
            original_pay, original_raw, _ = native.outcomes(x,groups)
            direct=focal_samples(x,rows)
            np.testing.assert_array_equal(original_pay*4,direct['payout4'])
            np.testing.assert_array_equal(np.rint(original_raw*12).astype(int),direct['castle12'])
            assert int(direct['payout4'].sum()) == round(scores[i,j]*4*n)
            checks+=n
            record[str(i)]={'payout_mean':float(direct['payout4'].mean()/4),
                'castle_mean':float(direct['castle12'].mean()/12),
                'standard_error':float(np.std(direct['payout4']/4,ddof=1)/math.sqrt(n)),
                'probability_zero':float(np.mean(direct['payout4']==0)),
                'sole_first':float(np.mean(direct['outright_first']))}
        ours=focal_samples(named['our_v1'],rows)['payout4']/4
        parry=focal_samples(named['parry_36_64'],rows)['payout4']/4
        d=parry-ours
        record['paired_parry_minus_ours']={'mean':float(d.mean()),'standard_error':float(d.std(ddof=1)/math.sqrt(n))}
        detail[field['name']]=record
        print('Independent full-group check',field['name'],flush=True)
    save(out/'engine-comparison.json',{'candidate_group_cells':checks,'fields':detail})
    # Retrospective stress scores on the unchanged 85 old saved banks, for the full new pool.
    old=read(ROOT/'results/game2-final/results.json'); old_scores=[]
    old_rows=[r['allocation'] for r in old['candidates'].values()]
    index={tuple(x):i for i,x in enumerate(actions.tolist())}
    old_ids=[index[tuple(x)]for x in old_rows]
    old_ids_names=list(old['candidates'])
    for key in old['cell_keys']:
        path=out/'old-stress'/(key+'.npz')
        if not path.exists():
            with np.load(ROOT/'results/game2-final/evaluation'/(key+'.npz'))as data:
                rows=data['triplets']; expected=data['rank_sum4']
            sums=exact_sample_sums(native,actions,native.compress_groups(rows),len(rows))
            np.testing.assert_array_equal(sums[old_ids],expected)
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,rank_sum4=sums,size=len(rows))
        with np.load(path)as data:
            old_scores.append(data['rank_sum4']/(4*int(data['size'])))
    old_scores=np.column_stack(old_scores)
    blocks={g:np.array([float(rational(w[key])) if key in w else 0.for key in old['cell_keys']])for g,w in old['policy']['block_weights'].items()}
    block_values={g:old_scores@w for g,w in blocks.items()}
    original_benchmarks={g:float(rational(v))for g,v in old['policy']['block_benchmarks'].items()}
    original_regret=np.maximum.reduce([original_benchmarks[g]-block_values[g]for g in blocks])
    np.savez_compressed(out/'old-stress-scores.npz',values=old_scores,names=old['cell_keys'],allocations=actions)
    save(out/'old-stress-summary.json',{'block_values':{g:v.tolist()for g,v in block_values.items()},
        'regret_to_original_benchmarks':original_regret.tolist(),
        'scope':'Known retrospective stresses. Original 191 score sums must match exactly. Original benchmarks and recommendations unchanged.'})
    print('Old stress banks agree exactly for all original 191 candidates',flush=True)
    # Reproduce original score-CDF validation, then repeat on independent own actions/opponents.
    public=read(vendor/'public-scores.json')['scores']
    cdf={}
    for phase in ['original','fresh']:
        per_family=config['cdf_'+phase+'_actions_per_family']
        action_seed=config['cdf_original_action_seed'] if phase=='original' else config['cdf_fresh_action_seed']
        own,families=sample_actions(catalog,per_family,action_seed)
        size=config['cdf_'+phase+'_groups']
        phase_results={}
        for j,field in enumerate(fields):
            path=out/('cdf-'+phase)/(field['name']+'.npz')
            seed=field['original_model']['validation_triplet_seed'] if phase=='original' else config['cdf_fresh_opponent_seed']+j
            if not path.exists():
                rows=draw_population(catalog,field['weights'],3*size,seed).reshape(size,3,5)
                u=native.evaluate(own,native.compress_groups(rows))
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,own_actions=own,own_families=families,opponent_triplets=rows,values=u,size=size,opponent_seed=seed,action_seed=action_seed)
            with np.load(path)as data:
                u=data['values'].copy()
            mass=np.array(field['weights'])[families]/per_family
            metrics=cdf_diagnostic(u,mass,public)
            thresholds=np.array(fit['thresholds'])
            coarse=np.array([mass[u<=t].sum()for t in thresholds])
            observed=np.array(fit['observed_cdf'])
            metrics['original_26_threshold_max_gap']=float(abs(coarse-observed).max())
            metrics['original_26_threshold_mean_gap']=float(abs(coarse-observed).mean())
            metrics['passes_original_26_threshold_screen']=metrics['original_26_threshold_max_gap']<=.085 and metrics['original_26_threshold_mean_gap']<=.035 and metrics['mass_above_public_max_plus_point_two']<=.015
            metrics['passes_full_curve_screen']=metrics['maximum_cdf_gap']<=.085 and metrics['mean_grid_cdf_gap']<=.035 and metrics['mass_above_public_max_plus_point_two']<=.015
            if phase=='original':
                row=field['original_model']
                metrics['max_absolute_difference_from_saved_cdf']=float(abs(coarse-row['validation_cdf']).max())
                metrics['difference_from_saved_population_mean']=float(mass@u-row['validation_population_mean_score'])
            phase_results[field['name']]=metrics
            print('CDF',phase,field['name'],'full gap',round(metrics['maximum_cdf_gap'],4),'seconds',round(time.perf_counter()-start,1),flush=True)
        cdf[phase]=phase_results
    save(out/'cdf-comparison.json',cdf)
    epsilon=9*math.sqrt(math.log(2*len(pool)*len(fields)/config['confidence_alpha'])/(2*n))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(freeze_path),
        'candidate_count':len(pool),'groups_per_field':n,'field_names':[f['name']for f in fields],
        'named_indices':ids,'ranking':ranking,'primary_catalog_leader':actions[selected].tolist(),
        'named_results':{name:{'allocation':actions[i].tolist(),'field_scores':dict(zip([f['name']for f in fields],scores[i].tolist())),
            'minimum':float(scores[i].min()),'maximum':float(scores[i].max()),
            'old_block_scores':{g:float(v[i])for g,v in block_values.items()},
            'old_maximum_regret':float(original_regret[i])}for name,i in ids.items()},
        'conditional_simultaneous_95_percent_cell_error_bound':epsilon,
        'conditional_maximin_shortfall_to_best_frozen_candidate_bound':2*epsilon,
        'confidence_scope':'Sampling only, jointly over this frozen finite candidate pool and these 17 unchanged laws. No bound on population modeling error or full-action optimality.',
        'independent_candidate_group_cells':checks,'original_candidates_exactly_matched_on_old_banks':191,
        'cdf_pass_counts':{phase:{rule:sum(r[rule]for r in rows.values())for rule in ['passes_original_26_threshold_screen','passes_full_curve_screen']}for phase,rows in cdf.items()},
        'elapsed_seconds':time.perf_counter()-start,'scope':config['scope']}
    for p in source_files+input_files:
        assert sha(p)==hashes[str(p.relative_to(ROOT))],str(p)
    save(out/'results.json',result)
    write_csv(out/'candidate-comparison.csv',['index','allocation','minimum_17','maximum_17','minimum_strict9','maximum_regret17','old_core','old_pressure','old_arenas','regret_to_old_benchmarks'],
        [[i,str(x),min(scores[i]),max(scores[i]),ranking['strict_9']['minimum'][i],ranking['all_17']['maximum_regret'][i],*[float(block_values[g][i])for g in ['core','pressure','arenas']],float(original_regret[i])]for i,x in enumerate(actions.tolist())])
    save(out/'run-complete.json',{'passed':True,'freeze_sha256':sha(freeze_path),'results_sha256':sha(out/'results.json'),'scope':'Runner assertions passed; separate verifier still required.'})
    print(json.dumps({'leader':result['primary_catalog_leader'],'named':result['named_results'],'cdf':result['cdf_pass_counts']},indent=2),flush=True)


if __name__=='__main__':
    main()
