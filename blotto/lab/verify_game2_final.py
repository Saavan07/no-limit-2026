"""Independent final scoring, deterministic arena replay, and selection audit."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from math import sqrt,log
import argparse,json,time,hashlib
import numpy as np
from blotto_lab.core import Population
from blotto_lab.multiwave import simple_json
from game2_lab.core import focal_samples,shared_triplets,reference_group,fixed_sample_bound
from game2_lab.human import payoff_arrays
from game2_lab.arena import class_name
from game2_lab.final import final_arena,prior_weights,core_triplets,pressure_triplets,evaluation_weights
from game2_lab.weight_counters import worst_clone_population
from run_game2_final import development_field,freeze_field,challenge_design,path_record
from run_game2_weights import definitions as weight_definitions,criteria
from run_multiwave_study import ROOT
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-final');out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');cfg=r['configuration'];m=read(out/'manifest.json');sig=read(out/'run-inputs.json');assert sig==m['signature']
    for key,digest in m['artifacts'].items():assert sha(out/key)==digest,key
    for key,digest in sig['source_hashes'].items():assert sha(ROOT/key)==sha(out/'source-snapshot'/key)==digest,key
    for key,digest in sig['input_hashes'].items():assert sha(ROOT/key)==digest,key
    freeze=read(out/'candidate-freeze.json');start=read(out/'challenge-start.json');evalstart=read(out/'evaluation-start.json');freeze_hash=sha(out/'candidate-freeze.json')
    assert freeze_hash==r['freeze_sha256']==m['freeze_sha256']==start['freeze_sha256']==evalstart['freeze_sha256']
    assert freeze['created_unix']<start['started_unix']<evalstart['started_unix']<m['completed_unix']
    assert freeze['selection_rule']==cfg['selection_rule']and freeze['candidates']==r['candidates']and freeze['comparison_tags']==r['comparison_tags']
    assert freeze['run_inputs_sha256']==sha(out/'run-inputs.json')and freeze['development_scores_sha256']==sha(out/'development-scores.npz')
    old=read(ROOT/'results/game2-weights/results.json');oldarena=read(ROOT/'results/game2-arena/results.json');oldcfg=old['configuration'];wd=weight_definitions(oldarena,oldcfg);actions,high,random=development_field(old,cfg);assert [list(x)for x in actions]==freeze['development_field']
    with np.load(out/'development-scores.npz')as a:values=a['rank_sum4']
    with np.load(ROOT/'results/game2-weights/development-scores.npz')as a:known=a['rank_sum4']
    reuse=cfg['development_triplets']==oldcfg['development_triplets'];nold=len(old['candidates'])
    for j,key in enumerate(wd['cells']):
        with np.load(ROOT/'results/game2-weights/development'/(key+'.npz'))as a:rows=a['triplets'][:cfg['development_triplets']]
        if reuse:
            np.testing.assert_array_equal(values[:nold,j],known[:,j]);actual=payoff_arrays(actions[nold:],rows,candidate_batch=11,triplet_batch=1701)[0].sum(axis=1,dtype=np.int64);np.testing.assert_array_equal(values[nold:,j],actual)
        else:np.testing.assert_array_equal(values[:,j],payoff_arrays(actions,rows)[0].sum(axis=1,dtype=np.int64))
    objective=criteria(values/(4*cfg['development_triplets']),wd,oldcfg);candidates,tags=freeze_field(actions,high,random,old,objective,cfg)
    assert simple_json(candidates)==r['candidates']and tags==r['comparison_tags']and simple_json(challenge_design(cfg))==r['design']==freeze['design']
    seeds=freeze['seed_pool'];cloud=freeze['counter_cloud'];steps=gates=gate_cells=0
    for spec in r['design']['arena_paths']:
        name=spec['name'];folder=out/'training'/name;assert read(folder/'result.json')==r['paths'][name]and sha(folder/'result.json')==evalstart['path_result_sha256'][name]
        state=read(folder/'checkpoint.json');stored=state.pop('state_sha256');assert hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()==stored
        with TemporaryDirectory(prefix='game2-final-audit-')as temp:replay=final_arena(temp,seeds,actions,cfg,spec['rule'],spec['prior'],spec['initial'],spec['seed'])
        assert simple_json(path_record(replay,spec,cfg))==r['paths'][name]
        with np.load(folder/'arrays.npz')as a:
            for key in ['final','average','tail']:np.testing.assert_array_equal(a[key],replay[key])
        assert replay['tail_count']==cfg['settling_steps']and replay['actions'][:len(seeds)]==[tuple(x)for x in seeds]
        prior=prior_weights(replay['actions'],spec['prior']);assert abs(prior.sum()-1)<1e-13 and np.all(prior>0)
        oldsize=len(seeds)
        for wave in replay['waves']:
            assert wave['step']<=cfg['steps']-cfg['settling_steps'];field=wave['field'];assert field[:oldsize]==[list(x)for x in replay['actions'][:oldsize]]
            for target in wave['targets'].values():
                pop=Population.from_dict(target['population']);rows=shared_triplets(pop,cfg['gate_triplets'],target['gate_seed'])['triplets'];a=focal_samples(target['winner'],rows)['payout4'];b=focal_samples(target['incumbent'],rows)['payout4'];gain=Fraction(int((a-b).sum()),4*len(rows));bound=fixed_sample_bound(len(rows),cfg['maximum_rank_gates'],cfg['gate_alpha'],paired=True)
                assert gain==rational(target['gate_mean_gain'])and bound==target['gate_bound']and target['gate_certified_improvement']==(float(gain)>bound)
                fitness=target['fitness_sums'];best=lambda allowed:min(allowed,key=lambda i:(-fitness[i],field[i]));assert target['winner']==field[best(range(len(field)))]and target['incumbent']==field[best(range(oldsize))]
                gates+=1;gate_cells+=2*len(rows)
            for admission in wave['admissions']:
                assert list(replay['actions'][admission['index']])==admission['allocation']
                if admission['is_new']:assert admission['index']==oldsize;oldsize+=1
        assert oldsize==len(replay['actions'])and all(v['selfplay_group_mean']==3.75 for v in replay['trace']);steps+=cfg['steps'];print('Replayed and gated final arena: '+name,flush=True)
    assert gates<=cfg['maximum_rank_gates']
    cids=list(r['candidates']);sums=[];castles=[];foundation_cells=triples=0
    for key in r['cell_keys']:
        cell=r['cells'][key]
        if cell['kind']=='core':expected=core_triplets(cell,cfg['evaluation_triplets'],cell['seed'],cloud)
        elif cell['kind']=='pressure':expected=pressure_triplets(cell,cfg['evaluation_triplets'],cell['seed'],cloud)
        else:expected=shared_triplets(Population.from_dict(cell['population']),cfg['evaluation_triplets'],cell['seed'])['triplets']
        with np.load(out/'evaluation'/(key+'.npz'))as a:rows=a['triplets'];rank=a['rank4'];rank_sum=a['rank_sum4'];castle_sum=a['castle_sum12']
        np.testing.assert_array_equal(rows,expected);assert np.all(rows.sum(axis=2)==100)and len(rows)==cfg['evaluation_triplets']
        for j,cid in enumerate(cids):
            direct=focal_samples(r['candidates'][cid]['allocation'],rows);np.testing.assert_array_equal(direct['payout4'],rank[j]);assert int(direct['payout4'].sum())==int(rank_sum[j])and int(direct['castle12'].sum())==int(castle_sum[j]);foundation_cells+=len(rows)
        sums.append(rank_sum);castles.append(castle_sum);triples+=len(rows);print('Every frozen candidate foundation-checked: '+key,flush=True)
    values=np.column_stack(sums)
    with np.load(out/'evaluation-sums.npz')as a:np.testing.assert_array_equal(values,a['rank_sum4']);np.testing.assert_array_equal(np.column_stack(castles),a['castle_sum12'])
    weights=evaluation_weights(r['design'],cfg);assert simple_json(weights)==r['policy']['block_weights'];cell_scores={cid:{key:Fraction(int(values[i,j]),4*cfg['evaluation_triplets'])for j,key in enumerate(r['cell_keys'])}for i,cid in enumerate(cids)}
    blocks={cid:{g:sum(w*cell_scores[cid][k]for k,w in group.items())for g,group in weights.items()}for cid in cids};maxima={g:max(v[g]for v in blocks.values())for g in weights}
    regrets={cid:{g:maxima[g]-v[g]for g in weights}for cid,v in blocks.items()};risk={cid:max(v.values())for cid,v in regrets.items()};mean={cid:sum(v.values())/3 for cid,v in blocks.items()}
    ranking=sorted(cids,key=lambda cid:(risk[cid],-mean[cid],r['candidates'][cid]['allocation']));p=r['policy']
    for key,value in [('block_scores',blocks),('block_benchmarks',maxima),('block_regrets',regrets),('maximum_block_regret',risk),('equal_block_mean',mean)]:assert simple_json(value)==p[key]
    assert ranking==p['ranking']and ranking[0]==p['selected_candidate_id']and r['selected_allocation']==r['candidates'][ranking[0]]['allocation']
    # Independently reconstruct every reported model score, sensitivity choice, and bound.
    from math import comb
    models={cid:{}for cid in cids}
    for ref,ids in r['design']['core_references'].items():
        for p0 in cfg['copy_points']:
            q=Fraction(str(p0));key=ref+'__p'+str(p0)
            for cid in cids:models[cid][key]=sum(comb(3,k)*q**k*(1-q)**(3-k)*cell_scores[cid][cell]for k,cell in enumerate(ids))
    for key,meta in p['model_definitions'].items():
        if meta['block']!='core':
            for cid in cids:models[cid][key]=cell_scores[cid][key]
    assert simple_json(models)==p['model_scores'];minimum={cid:min(v.values())for cid,v in models.items()};assert simple_json(minimum)==p['lowest_model_score']
    low={cid:sum(v for k,v in rows.items()if p['model_definitions'][k]['block']=='core'and rational(p['model_definitions'][k]['copy_share'])<=Fraction(1,10))/(2*len(r['design']['core_references']))for cid,rows in models.items()}
    assert simple_json(low)==p['low_copy_core_mean'];choose=lambda value:min(cids,key=lambda cid:(-value[cid],-mean[cid],r['candidates'][cid]['allocation']))
    assert p['alternatives']=={'equal_block_mean':choose(mean),'lowest_model_score':choose(minimum),'low_copy_core_mean':choose(low)}
    alpha=cfg['sampling_alpha']/2;cell_bound=fixed_sample_bound(cfg['evaluation_triplets'],len(cids)*len(r['cells']),alpha);assert cell_bound==r['cell_error_bound']
    for group,w in weights.items():assert r['block_error_bounds'][group]==9*sqrt(log(2*len(cids)*3/alpha)/2*sum(float(v)**2/cfg['evaluation_triplets']for v in w.values()))
    homogeneous=0
    for cid,case in r['homogeneous_stress'].items():
        x=r['candidates'][cid]['allocation'];actual=worst_clone_population(x);assert simple_json(actual)==case['exact_worst_clone']
        for y,score in case['known_pure_populations']:assert reference_group([x,y,y,y]).payouts[0]==rational(score);homogeneous+=1
    for key,w in r['response_witnesses'].items():
        score=max(cell_scores[cid][key]for cid in cids);best=min((cid for cid in cids if cell_scores[cid][key]==score),key=lambda cid:r['candidates'][cid]['allocation']);assert w['candidate_id']==best and rational(w['score'])==score and rational(w['gap_above_selfplay'])==score-Fraction(15,4)and w['lower_gap']==max(0,float(score-Fraction(15,4))-cell_bound)
    for filename in ['game1-selected-v1.json','game2-foundation-checkpoint-v1.json','game2-human-checkpoint-v1.json','game2-arena-checkpoint-v1.json','game2-weights-checkpoint-v1.json']:
        for key,digest in read(ROOT/'results'/filename)['source_artifact_sha256'].items():assert sha(ROOT/key)==digest,key
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
        'development_field':len(actions),'frozen_candidates':len(cids),'arena_paths_replayed':len(r['paths']),'arena_steps_replayed':steps,'independent_rank_gates':gates,'gate_candidate_group_cells_foundation_checked':gate_cells,
        'evaluation_conditional_cells':len(r['cells']),'iid_model_comparisons':len(p['model_definitions']),'legal_evaluation_triplets_regenerated':triples,'all_frozen_candidate_group_cells_foundation_checked':foundation_cells,
        'exact_homogeneous_stress_groups':homogeneous,'inverse_population_oracles_reproduced':len(cids),'selection_rule_independently_recomputed':True,'recommendation':r['selected_allocation'],'earlier_checkpoints_unchanged':True,'freeze_sha256':freeze_hash,
        'scope':'Verified frozen final challenge and conditional recommendation; no applicant forecast, global optimum, or Nash certificate'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
