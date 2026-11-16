"""Replay Game 2 arenas and independently audit gates, selection, and scoring."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from math import comb
import argparse,hashlib,json,time
import numpy as np
from blotto_lab.core import Population,iter_allocations,validate_allocation
from blotto_lab.multiwave import simple_json
from game2_lab.core import shared_triplets,focal_samples,group_scores12_4,fixed_sample_bound
from game2_lab.arena import arena_path,initial_shares,as_population,metric_arrays,static_triplets,class_name,single_triplet_response
from run_game2_arena import path_definitions,controls,snapshots_record
from run_multiwave_study import ROOT
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def independent_metric_sums(actions,rows):
    result=[]
    for x in actions:
        totals=np.zeros(3,dtype=np.int64)
        for start in range(0,len(rows),8192):
            expected=focal_samples(x,rows[start:start+8192])
            totals+=np.array([expected['payout4'].sum(),expected['castle12'].sum(),expected['finish_credits12'][:,0].sum()],dtype=np.int64)
        result.append(totals)
    return np.array(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-arena')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');m=read(out/'manifest.json');cfg=r['configuration']
    for name,digest in m['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in m['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in m['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    assert m['signature']==read(out/'run-inputs.json')
    freeze=read(out/'post-training-freeze.json');start=read(out/'evaluation-start.json');marker=read(out/'training-start.json');freeze_hash=sha(out/'post-training-freeze.json')
    assert freeze_hash==r['freeze_sha256']==m['freeze_sha256']==start['freeze_sha256']
    assert marker['started_unix']<freeze['created_unix']<start['started_unix']<m['completed_unix']
    for key in ['candidates','primary','shortlist']:assert freeze[key]==r[key]
    for file,key in [('selection-definitions.json','selection_definitions_sha256'),('selection-scores.npz','selection_scores_sha256'),('run-inputs.json','run_inputs_sha256'),('training-start.json','training_start_sha256')]:assert sha(out/file)==freeze[key]
    assert sha(out/'seed-pool.json')==marker['seed_pool_sha256']and sha(out/'counter-cloud.json')==marker['cloud_sha256']
    assert marker['path_definitions']==path_definitions(cfg)
    expected_static=[{'key':family+'_'+str(v),'family':family,'variant':v,'size':cfg['evaluation_triplets'],'seed':cfg['seed']+50000+100*j+v}for j,family in enumerate(cfg['static_families'])for v in [0,1]]
    assert expected_static==marker['static_definitions']
    old=read(ROOT/'results/game2-human/results.json');archive=[tuple(v['allocation'])for v in old['candidates'].values()]
    seed_pool=[tuple(x)for x in read(out/'seed-pool.json')];cloud=[tuple(x)for x in read(out/'counter-cloud.json')]
    gate_count=gate_cells=replay_steps=oracle_count=0
    for definition in marker['path_definitions']:
        name=definition['name'];folder=out/'training'/name;p=r['paths'][name]
        assert sha(folder/'result.json')==freeze['path_result_sha256'][name]and read(folder/'result.json')==p
        checkpoint=read(folder/'checkpoint.json');stored=checkpoint.pop('state_sha256')
        assert hashlib.sha256(json.dumps(checkpoint,sort_keys=True,separators=(',',':')).encode()).hexdigest()==stored
        assert checkpoint['step']==cfg['steps']and checkpoint['tail_count']==cfg['settling_steps']
        initial=initial_shares(seed_pool,definition['initial'],definition['initial_seed'])
        with TemporaryDirectory(prefix='game2-arena-audit-')as temp:
            replay=arena_path(temp,seed_pool,archive,initial,definition['type_shares'],definition['rule'],definition['history_mass'],definition['mutation'],cfg,definition['path_seed'])
        assert simple_json(snapshots_record(replay,definition,cfg))==p
        with np.load(folder/'arrays.npz')as a:
            np.testing.assert_array_equal(a['initial'],initial)
            for key in ['types','final','average','tail_average']:np.testing.assert_array_equal(a[key],replay[key])
        assert np.allclose(replay['types'].sum(axis=1),1,atol=1e-13)
        assert replay['actions'][:len(seed_pool)]==seed_pool and len(set(replay['actions']))==len(replay['actions'])
        last_size=len(seed_pool)
        for wave in p['waves']:
            assert wave['step']<=cfg['steps']-cfg['settling_steps']and wave['step']%cfg['innovation_every']==0
            field=[tuple(x)for x in wave['field']];assert field[:last_size]==replay['actions'][:last_size]
            for case in wave['fixed_triplet_oracles']:
                response=single_triplet_response(case['opponents']);assert list(response['allocation'])==case['response']['allocation']
                score=focal_samples(response['allocation'],np.array([case['opponents']]))['payout4'][0]
                assert Fraction(int(score),4)==rational(case['response']['payout']);oracle_count+=1
            for target in wave['targets'].values():
                pop=Population.from_dict(target['population']);rows=shared_triplets(pop,cfg['gate_triplets'],target['gate_seed'])['triplets']
                choices=[tuple(target[k])for k in ['rank_winner','incumbent']];actual=independent_metric_sums(choices,rows)
                gain=Fraction(int(actual[0,0]-actual[1,0]),4*len(rows));bound=fixed_sample_bound(len(rows),cfg['maximum_rank_gates'],cfg['gate_alpha'],paired=True)
                assert gain==rational(target['gate_mean_gain'])and abs(bound-target['gate_bound'])<1e-14
                assert target['gate_certified_improvement']==(float(gain)>bound)
                sums=np.array(target['fitness_sums']);best=lambda allowed:min(allowed,key=lambda j:(-int(sums[0,j]),field[j]))
                assert choices[0]==field[best(range(len(field)))]and choices[1]==field[best(range(last_size))]
                gate_count+=1;gate_cells+=2*len(rows)
            for admission in wave['admissions']:
                x=validate_allocation(admission['allocation']);assert replay['actions'][admission['index']]==x
                if admission['is_new']:assert admission['index']==last_size;last_size+=1
            assert all(t['selfplay_group_mean']==3.75 and t['type_shares']==definition['type_shares']for t in p['trace'])
        assert last_size==len(replay['actions'])
        replay_steps+=cfg['steps'];print('Replayed and independently gated: '+name,flush=True)
    assert gate_count<=cfg['maximum_rank_gates']
    # Recompute the entire finite selection matrix from saved and regenerated legal triples.
    actions=[tuple(v['allocation'])for v in r['candidates'].values()];cids=list(r['candidates']);d=read(out/'selection-definitions.json');sums=[]
    for x in actions:validate_allocation(x)
    with np.load(out/'selection-scores.npz')as a:saved_values=a['rank_sum4']
    for key in d['keys']:
        model=d['models'][key];expected=shared_triplets(Population.from_dict(model['population']),cfg['selection_triplets'],model['seed'])['triplets']
        with np.load(out/'selection'/(key+'.npz'))as a:rows=a['triplets'];np.testing.assert_array_equal(rows,expected)
        sums.append(metric_arrays(actions,rows,candidate_batch=11,triplet_batch=1703)[0].sum(axis=1,dtype=np.int64))
    values=np.column_stack(sums);np.testing.assert_array_equal(values,saved_values);ri=d['rank_indices'];ci=d['static_center_index']
    objectives={'rank_tail_mean':values[:,ri].sum(axis=1),'rank_tail_minimum':values[:,ri].min(axis=1),
                'all_selection_regret':(values-values.max(axis=0)[None,:]).min(axis=1),'half_static_half_rank':8*values[:,ci]+values[:,ri].sum(axis=1)}
    for name,value in objectives.items():
        chosen=min(range(len(actions)),key=lambda j:(-int(value[j]),actions[j]));p=r['primary'][name]
        assert p['candidate_id']==cids[chosen]and tuple(p['allocation'])==actions[chosen]and p['objective_numerator']==int(value[chosen])
    for name,x in controls(old).items():assert tuple(r['primary'][name]['allocation'])==x
    print('Complete finite-field selection matrix and four rules verified.',flush=True)
    unique_primary=list(dict.fromkeys(p['candidate_id']for p in r['primary'].values()));eval_triplets=independent_cells=0
    eval_actions=[r['candidates'][cid]['allocation']for cid in r['shortlist']]
    for key,model in r['models'].items():
        if model['kind']=='trained_arena':expected=shared_triplets(Population.from_dict(model['population']),cfg['evaluation_triplets'],model['seed'])['triplets']
        else:
            definition=model['definition'];assert definition in expected_static
            expected=static_triplets(definition['family'],definition['variant'],definition['size'],definition['seed'],cloud)
        with np.load(out/model['file'])as a:rows=a['triplets'];rank=a['rank4'];castles=a['castle_sum12'];firsts=a['first_sum12']
        np.testing.assert_array_equal(rows,expected);assert np.all(rows.sum(axis=2)==100)
        calculated=metric_arrays(eval_actions,rows,candidate_batch=13,triplet_batch=7001)
        np.testing.assert_array_equal(calculated[0],rank)
        np.testing.assert_array_equal(calculated[1].sum(axis=1,dtype=np.int64),castles)
        np.testing.assert_array_equal(calculated[2].sum(axis=1,dtype=np.int64),firsts)
        for j,cid in enumerate(r['shortlist']):
            assert Fraction(int(rank[j].sum(dtype=np.int64)),4*len(rows))==rational(r['scores'][cid][key])
            assert Fraction(int(castles[j]),12*len(rows))==rational(r['castle_scores'][cid][key])
            assert Fraction(int(firsts[j]),12*len(rows))==rational(r['first_credits'][cid][key])
        direct=independent_metric_sums([r['candidates'][cid]['allocation']for cid in unique_primary],rows)
        for i,cid in enumerate(unique_primary):
            j=r['shortlist'].index(cid);np.testing.assert_array_equal(direct[i],[rank[j].sum(dtype=np.int64),castles[j],firsts[j]])
        eval_triplets+=len(rows);independent_cells+=len(rows)*len(unique_primary)
        print('Verified frozen evaluation: '+key,flush=True)
    bound=fixed_sample_bound(cfg['evaluation_triplets'],len(r['shortlist'])*len(r['models']),cfg['sampling_alpha']);assert abs(bound-r['sampling_bound'])<1e-14
    for cid,by_model in r['scores'].items():
        for group,stats in r['summary'][cid].items():
            keys=[k for k,v in r['models'].items()if v['group']==group];scores=[rational(by_model[k])for k in keys]
            assert rational(stats['mean'])==sum(scores)/len(scores)and rational(stats['minimum'])==min(scores)and rational(stats['maximum'])==max(scores)
            assert stats['minimum_scenarios']==[k for k in keys if rational(by_model[k])==min(scores)]
    for key,w in r['response_witnesses'].items():
        scores={cid:rational(v[key])for cid,v in r['scores'].items()};value=max(scores.values());cid=min((c for c in scores if scores[c]==value),key=lambda c:r['candidates'][c]['allocation'])
        assert w['candidate_id']==cid and rational(w['estimated_response_score'])==value and rational(w['estimated_gap_above_selfplay'])==value-Fraction(15,4)
        assert abs(w['simultaneous_lower_gap']-max(0,float(value-Fraction(15,4))-bound))<1e-14
    # Independent foundation scan of every legal allocation for the coordinated-group ceiling.
    opponents=np.array(r['worst_group_benchmark']['opponents']);iterator=iter_allocations();count=ties=0;best=-1;best_action=None
    while True:
        batch=list(islice(iterator,32768))
        if not batch:break
        groups=np.concatenate((np.array(batch)[:,None,:],np.broadcast_to(opponents,(len(batch),3,5))),axis=1)
        _,payout=group_scores12_4(groups);scores=payout[:,0];maximum=int(scores.max())
        if maximum>best:best=maximum;ties=0;best_action=None
        if maximum==best:
            positions=np.flatnonzero(scores==best);ties+=len(positions);candidate=min(batch[int(j)]for j in positions)
            best_action=candidate if best_action is None else min(best_action,candidate)
        count+=len(batch)
    assert count==comb(104,4)==4598126 and best==4 and ties==comb(102,4)+3
    oracle=r['worst_group_benchmark']['exact_response'];assert rational(oracle['payout'])==1 and tuple(oracle['allocation'])==best_action
    for name in ['game1-selected-v1.json','game2-foundation-checkpoint-v1.json','game2-human-checkpoint-v1.json']:
        for path,digest in read(ROOT/'results'/name)['source_artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
       'paths_replayed':len(r['paths']),'training_steps_replayed':replay_steps,'independent_rank_gates':gate_count,'gate_candidate_triplet_cells_foundation_checked':gate_cells,
       'fixed_triplet_oracles_reproduced':oracle_count,'finite_candidate_field':len(actions),'selection_scenarios':len(d['keys']),
       'frozen_primary_rules':len(r['primary']),'unique_primary_allocations':len(unique_primary),'frozen_evaluation_candidates':len(r['shortlist']),
       'evaluation_scenarios':len(r['models']),'evaluation_shared_triplets_regenerated':eval_triplets,'primary_evaluation_candidate_triplet_cells_foundation_checked':independent_cells,
       'worst_group_all_allocations_scanned':count,'worst_group_maximum_payout':Fraction(best,4),'worst_group_optimum_count':ties,'worst_group_lexicographic_optimum':best_action,
       'sampling_bound':bound,'freeze_sha256':freeze_hash,'earlier_checkpoints_unchanged':True,
       'scope':'Sampled-fitness adapting arenas and finite frozen comparisons; no Nash certificate or applicant forecast'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2,default=str))


if __name__=='__main__':main()
