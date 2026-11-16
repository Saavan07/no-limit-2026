"""Independent frozen-selection, generator, coupled-learning, and all-action audit."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import hashlib
import json
import time
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population,best_response,iter_allocations,mix_tables,score,security_of
from blotto_lab.contamination import FrontierObjective,pattern_security_twice
from blotto_lab.historical import rows_to_table,sampled_metadata,validate_rows
from blotto_lab.merged import coupled_path,mixture_weights,quantize,shifted_rows
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def same_table(a,b):
    assert a.game==b.game
    for ar,br in zip(a.numerators,b.numerators):assert all(x*b.denominator==y*a.denominator for x,y in zip(ar,br))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/merged-game1')
    out=parser.parse_args().study.resolve();clock=time.perf_counter()
    r,m=read(out/'results.json'),read(out/'manifest.json');signature=m['signature'];cfg=r['configuration']
    for path,digest in m['artifacts'].items():assert sha(out/path)==digest,path
    for path,digest in signature['source_hashes'].items():assert sha(ROOT/path)==sha(out/'source-snapshot'/path)==digest,path
    for path,digest in signature['input_hashes'].items():assert sha(ROOT/path)==digest,path
    assert signature==read(out/'run-inputs.json')
    freeze=read(out/'frozen-candidates.json');marker=read(out/'evaluation-start.json')
    assert sha(out/'frozen-candidates.json')==r['freeze_sha256']==m['freeze_sha256']==marker['freeze_sha256']
    assert freeze['created_unix']==r['frozen_unix']<r['evaluation_started_unix']==marker['started_unix']
    for file,field in [('development-score-tables.json','selection_inputs_sha256'),('run-inputs.json','run_inputs_sha256'),('seed-pool.json','seed_pool_sha256')]:assert sha(out/file)==freeze[field]
    for key in ['candidates','primary','label_ids','development_keys']:assert freeze[key]==r[key]
    defs=read(out/'development-score-tables.json');search_keys=defs['search_keys']
    tables=[unpack_table(defs['tables'][k],k) for k in search_keys]
    component_order=['machine_behavior','human_static','old_strategic','point_learners','win_learners']
    components={k:tables[defs['components'][k]] for k in component_order}
    machine=read(ROOT/'results/contamination-study/results.json');human=read(ROOT/'results/human-challenge/results.json');broad=read(ROOT/'results/human-scout-stress/results.json')
    same_table(components['machine_behavior'],unpack_table(machine['reference_models']['ref_balanced_s0'],'M'))
    same_table(components['old_strategic'],unpack_table(machine['reference_models']['ref_balanced_s1'],'S'))
    human_refs=[]
    for name in ['all_open','all_heavy5']:
        human_refs.append(unpack_table(human['references'][name],name));keys=broad['references'][name]['model_keys']
        human_refs.append(mix_tables([unpack_table(broad['models'][k],k) for k in keys],[1]*len(keys)))
    same_table(components['human_static'],mix_tables(human_refs,[1]*4))
    for fitness,component in [('points','point_learners'),('wins','win_learners')]:
        source=[Population.from_dict(p[s]['population']).table() for p in human['learning'].values() if p['fitness']==fitness for s in ['final','average']]
        same_table(components[component],mix_tables(source,[1]*len(source)))
    for corner in defs['corners'].values():
        weights=mixture_weights(*(corner['parameters'][k] for k in ['h','l','w','z']))
        assert list(weights)==[rational(v) for v in corner['weights']]
        same_table(tables[corner['table_index']],mix_tables(list(components.values()),weights))
    weights=mixture_weights(*(cfg['midpoint_prior'][k] for k in ['h','l','w','z']))
    same_table(tables[defs['midpoint']['table_index']],mix_tables(list(components.values()),weights))
    all_tables={k:unpack_table(model,k) for k,model in r['models'].items()};optima_all={k:best_response(t) for k,t in all_tables.items()}
    sampled=0
    for key,model in r['models'].items():
        br=optima_all[key]
        assert list(br.allocation)==model['best_response']['allocation']
        assert br.expected_score==rational(model['best_response']['score']) and br.optimum_count==model['best_response']['optimum_count']
        if model['phase']=='development':same_table(all_tables[key],tables[search_keys.index(key)])
        if model['kind']=='fresh_generator':
            with np.load(out/model['sample_file']) as a:rows=a['allocations']
            validate_rows(rows);regenerated,_=shifted_rows(model['definition'],model['sample_size'],model['seed'])
            np.testing.assert_array_equal(rows,regenerated)
            assert sampled_metadata(rows,model['seed'])['draw_sha256']==model['draw_sha256']
            same_table(all_tables[key],rows_to_table(rows,key));sampled+=1
    seed_actions=[tuple(x) for x in read(out/'seed-pool.json')]
    assert seed_actions==[tuple(x) for x in human['pool_actions']]
    learner_tables=[];learner_records=[];innovation_checks=0;maximum_replay_error=0
    with TemporaryDirectory() as temp:
        for name,p in r['learning'].items():
            folder=out/'learning'/name
            with np.load(folder/'arrays.npz') as a:initial=a['initial'];point=a['point'];win=a['win'];final=a['final'];average=a['average']
            replay=coupled_path(Path(temp)/name,seed_actions,initial,p['win_share'],p['rule'],p['mutation'],cfg,p['seed'])
            assert replay['actions']==[tuple(x) for x in p['actions']]
            for field,stored in [('point',point),('win',win),('final',final),('average',average)]:
                error=float(np.abs(replay[field]-stored).max());maximum_replay_error=max(maximum_replay_error,error)
                np.testing.assert_allclose(replay[field],stored,rtol=0,atol=1e-13)
            np.testing.assert_allclose(final,(1-p['win_share'])*point+p['win_share']*win,rtol=0,atol=1e-14)
            assert [tuple(x) for x in p['actions'][:len(seed_actions)]]==seed_actions
            for trace in p['trace']:
                assert trace['fixed_win_share']==p['win_share'] and abs(trace['official_population_mean']-7.5)<1e-10
            for event in p['innovations']:
                population=Population.from_dict(event['pre_invasion_population']);br=best_response(population.table())
                assert list(br.allocation)==event['response']['allocation']
                assert br.expected_score==rational(event['response']['expected_score'])
                innovation_checks+=1
            state=read(folder/'checkpoint.json');state_digest=state.pop('state_sha256')
            assert hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()==state_digest
            assert state['step']==cfg['learning_steps'] and state['input_digest']==p['input_digest']
            for snapshot,array in [('final',final),('average',average)]:
                pop=Population.from_dict(p[snapshot]['population']);masses=quantize(array,cfg['quantization_mass'])
                assert pop.entries==Population.from_counts([(x,v) for x,v in zip(p['actions'],masses) if v]).entries
                exact=security_of(pop)
                assert exact['exploitability']==rational(p[snapshot]['certificate']['exploitability'])
                assert exact['security']==rational(p[snapshot]['certificate']['security'])
                key=name+'_'+snapshot;same_table(all_tables[key],pop.table())
                learner_tables.append(pop.table());learner_records.append(p[snapshot]['certificate']['best_response'])
            print(f'Coupled path reproduced: {name}',flush=True)
    counts={int(k):v for k,v in r['class_counts'].items()};first={int(k):tuple(v) for k,v in r['class_first'].items()}
    frontiers={name:{int(c):p for c,p in points.items()} for name,points in r['frontiers'].items()}
    objectives=[FrontierObjective(o['name'],tuple(o['indices']),o['mode']) for o in defs['objectives']]
    optimum=[best_response(t).expected_score for t in tables];scale=10**9
    upper=np.array([[[(n*scale+t.denominator-1)//t.denominator for n in row] for row in t.numerators] for t in tables],dtype=np.int64)
    lower_opt=np.array([v.numerator*scale//v.denominator for v in optimum],dtype=np.int64)
    thresholds=[]
    for o in objectives:
        thresholds.append(np.array([-( -rational(frontiers[o.name][c]['nominal_value']).numerator*scale//rational(frontiers[o.name][c]['nominal_value']).denominator) if c in counts else np.iinfo(np.int64).max for c in range(31)],dtype=np.int64))
    tied={o.name:{c:0 for c in counts} for o in objectives};winner={o.name:{c:None for c in counts} for o in objectives}
    with np.load(out/'security-classes.npz') as a:saved_security=a['security_twice']
    lr=np.array([t.numerators for t in learner_tables],dtype=np.int64)
    assert all(sum(max(row) for row in t.numerators)<np.iinfo(np.int64).max for t in learner_tables)
    lbest=[None]*len(lr);lties=[0]*len(lr);lwinner=[None]*len(lr)
    actual_counts={c:0 for c in counts};actual_first={};iterator=iter_allocations();done=0
    while True:
        batch=list(islice(iterator,cfg['verification_batch_size']))
        if not batch:break
        actions=np.asarray(batch,dtype=np.uint8);security=pattern_security_twice(actions)
        np.testing.assert_array_equal(security,saved_security[done:done+len(batch)])
        for c in np.unique(security):
            c=int(c);positions=np.flatnonzero(security==c);actual_counts[c]+=len(positions);actual_first.setdefault(c,tuple(map(int,actions[positions[0]])))
        score_upper=np.zeros((len(tables),len(batch)),dtype=np.int64);learner_scores=np.zeros((len(lr),len(batch)),dtype=np.int64)
        for i in range(5):score_upper+=upper[:,i,actions[:,i]];learner_scores+=lr[:,i,actions[:,i]]
        for j,o in enumerate(objectives):
            selected=score_upper[list(o.indices)]
            if o.mode=='regret':selected=selected-lower_opt[list(o.indices),None]
            possible=np.flatnonzero(selected.min(axis=0)>=thresholds[j][security])
            for index in possible:
                c=int(security[index]);x=tuple(map(int,actions[index]));values=[tables[i].expected_score(x) for i in o.indices]
                if o.mode=='regret':values=[v-optimum[i] for v,i in zip(values,o.indices)]
                v=min(values);expected=rational(frontiers[o.name][c]['nominal_value']);assert v<=expected,(o.name,c,x)
                if v==expected:
                    tied[o.name][c]+=1;old=winner[o.name][c];winner[o.name][c]=x if old is None else min(old,x)
        for j,values in enumerate(learner_scores):
            best=int(values.max());positions=np.flatnonzero(values==best)
            if lbest[j] is None or best>lbest[j]:lbest[j]=best;lties[j]=0;lwinner[j]=tuple(map(int,actions[positions[0]]))
            if best==lbest[j]:lties[j]+=len(positions)
        done+=len(batch)
        if done%1000000<cfg['verification_batch_size'] or done==DEFAULT_GAME.actions:print(f'Independent all-action audit {done:,}/{DEFAULT_GAME.actions:,}',flush=True)
    assert actual_counts==counts and actual_first==first
    for o in objectives:
        for c,p in frontiers[o.name].items():
            assert tuple(p['allocation'])==winner[o.name][c] and p['optimum_count']==tied[o.name][c],(o.name,c)
    for j,table in enumerate(learner_tables):
        assert Fraction(lbest[j],table.denominator)==rational(learner_records[j]['expected_score'])
        assert list(lwinner[j])==learner_records[j]['allocation'] and lties[j]==learner_records[j]['optimum_count']
    for primary in r['primary'].values():
        allowed=[p for c,p in frontiers[primary['objective']].items() if Fraction(c,2)>=primary['security_floor']]
        maximum=max(rational(p['nominal_value']) for p in allowed);wins=[p for p in allowed if rational(p['nominal_value'])==maximum]
        assert rational(primary['value'])==maximum
        assert tuple(primary['allocation'])==min(tuple(p['allocation']) for p in wins)
        assert primary['optimum_count']==sum(p['optimum_count'] for p in wins)
    cells=0
    for cid,candidate in r['candidates'].items():
        x=candidate['allocation'];guarantee=rational(r['security'][cid]['guarantee'])
        assert score(x,r['security'][cid]['counter'])==guarantee==security_of(Population.from_people([tuple(x)]))['security']
        for key,table in all_tables.items():
            v=table.expected_score(x);assert v==rational(r['scores'][cid][key])
            assert optima_all[key].expected_score-v==rational(r['regrets'][cid][key]);cells+=1
        for phase in ['development','validation','audit']:
            keys=[k for k,m in r['models'].items() if m['phase']==phase]
            assert min(rational(r['scores'][cid][k]) for k in keys)==rational(r['summary'][cid][phase]['minimum'])
            assert max(rational(r['regrets'][cid][k]) for k in keys)==rational(r['summary'][cid][phase]['worst_regret'])
        for name,corner in {**defs['corners'],'midpoint':defs['midpoint']}.items():assert tables[corner['table_index']].expected_score(x)==rational(r['prior_scores'][cid][name])
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
        'frozen_primary_selections':len(r['primary']),'frozen_candidates':len(r['candidates']),'scenario_tables':len(all_tables),
        'sampled_populations_regenerated':sampled,'legal_draws_checked':sampled*cfg['evaluation_sample_size'],
        'coupled_paths_reproduced':len(r['learning']),'maximum_replay_absolute_error':maximum_replay_error,
        'global_point_innovations_checked':innovation_checks,'all_actions_independently_checked':done,
        'conditional_global_optima_and_tie_counts':len(objectives)*len(counts),'learning_certificates_independently_enumerated':len(learner_tables),
        'score_regret_cells_checked':cells,'complete_parameter_corner_construction_verified':True,'freeze_sha256':r['freeze_sha256'],
        'scope':'Exact conditional-model and fresh constructed-challenge verification, not applicant-population inference'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
