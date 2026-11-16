"""Independent final-challenge accounting, replay, and integer global-selection audit."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import hashlib
import json
import time
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population,best_response,iter_allocations,mix_tables,score
from blotto_lab.contamination import pattern_security_twice
from blotto_lab.historical import rows_to_table,sampled_metadata,validate_rows
from final_tools import final_rows,enhanced_path,population_from_shares
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha
from verify_failure_analysis import forward_response
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def equal_table(a,b):
    assert a.game==b.game
    for ar,br in zip(a.numerators,b.numerators):assert all(x*b.denominator==y*a.denominator for x,y in zip(ar,br))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/final-game1')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');m=read(out/'manifest.json');cfg=r['configuration']
    for name,digest in m['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in m['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in m['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    assert read(out/'run-inputs.json')==m['signature']
    f=read(out/'frozen-candidates.json');start=read(out/'evaluation-start.json')
    assert f['created_unix']<start['started_unix'] and sha(out/'frozen-candidates.json')==start['freeze_sha256']==r['freeze_sha256']==m['freeze_sha256']
    assert f['candidates']==r['candidates']and f['primary']==r['primary']
    for file,key in [('selection-tables.json','selection_tables_sha256'),('selection-frontiers.json','selection_frontiers_sha256'),('arena-seed-pool.json','seed_pool_sha256'),('run-inputs.json','run_inputs_sha256')]:assert sha(out/file)==f[key]
    selections=read(out/'selection-tables.json');front=read(out/'selection-frontiers.json');names=selections['names'];st=[unpack_table(selections['models'][k],k)for k in names]
    old=read(ROOT/'results/merged-game1/results.json');olddefs=read(ROOT/'results/merged-game1/development-score-tables.json');diagnostic=read(ROOT/'results/failure-analysis/results.json')
    base=unpack_table(olddefs['tables'][olddefs['search_keys'][olddefs['midpoint']['table_index']]],'base');equal_table(st[0],base)
    assert selections['critical_keys']==list(diagnostic['critical_scenarios'])
    for raw in cfg['replacement_caps']:
        for k in selections['critical_keys']:
            t=unpack_table(old['models'][k],k);equal_table(st[names.index('cap'+raw+'__'+k)],mix_tables([base,t],[1-Fraction(raw),Fraction(raw)]))
    kinds={}
    for k in selections['critical_keys']:kinds.setdefault(old['models'][k]['kind'],[]).append(unpack_table(old['models'][k],k))
    q=mix_tables([mix_tables(v,[1]*len(v))for v in kinds.values()],[1]*len(kinds));h=Fraction(cfg['mean_hedge_share'])
    equal_table(st[-1],mix_tables([base,q],[1-h,h]))
    # Original primary vectors and all prior candidates survive unchanged.
    for name,p in old['primary'].items():assert r['primary'][name]['allocation']==p['allocation']
    actual_actions={tuple(c['allocation'])for c in r['candidates'].values()}
    assert all(tuple(c['allocation'])in actual_actions for c in old['candidates'].values())
    for data in diagnostic['maps'].values():
        for label in ['unrestricted','security2']:
            assert all(tuple(s['allocation'])in actual_actions for s in data[label]['segments']+data[label]['boundaries'])
    all_tables={k:unpack_table(model,k)for k,model in r['models'].items()};sampled=0;sample_rows=0
    for k,model in r['models'].items():
        table=all_tables[k];br=forward_response(table)
        assert list(br.allocation)==model['best_response']['allocation']and br.expected_score==rational(model['best_response']['expected_score'])and br.optimum_count==model['best_response']['optimum_count']
        if model['kind']=='fresh_static':
            with np.load(out/model['sample_file'])as a:rows=a['allocations']
            validate_rows(rows);regenerated=final_rows(model['definition'],model['sample_size'],model['seed']);np.testing.assert_array_equal(rows,regenerated)
            assert sampled_metadata(rows,model['seed'])['draw_sha256']==model['draw_sha256'];equal_table(table,rows_to_table(rows,k));sampled+=1;sample_rows+=len(rows)
        if 'component_keys'in model:
            weights=[rational(v)if isinstance(v,dict)else Fraction(v)for v in model['component_weights']]
            equal_table(table,mix_tables([all_tables[key]for key in model['component_keys']],weights))
    assert sampled==52 and sample_rows==10400000
    seed_pool=[tuple(x)for x in read(out/'arena-seed-pool.json')];human=read(ROOT/'results/human-challenge/results.json')
    assert seed_pool[:len(human['pool_actions'])]==[tuple(x)for x in human['pool_actions']]
    learner_tables=[];learner_records=[];events=mutants=admitted=0;maximum_replay_error=0
    with TemporaryDirectory()as temp:
        for name,p in r['learning'].items():
            folder=out/'learning'/name
            with np.load(folder/'arrays.npz')as a:initial=a['initial'];point=a['point'];win=a['win'];final=a['final'];average=a['average']
            rng=np.random.default_rng(p['initial_seed']);np.testing.assert_array_equal(initial,rng.dirichlet(np.full(len(seed_pool),2 if p['phase']=='challenge'else .08)))
            replay=enhanced_path(Path(temp)/name,seed_pool,initial,p['win_share'],p['rule'],p['mutation'],cfg,p['path_seed'])
            assert replay['actions']==[tuple(x)for x in p['actions']]
            for key,stored in [('point',point),('win',win),('final',final),('average',average)]:
                err=float(np.abs(replay[key]-stored).max());maximum_replay_error=max(maximum_replay_error,err);np.testing.assert_allclose(replay[key],stored,rtol=0,atol=1e-13)
            np.testing.assert_allclose(final,(1-p['win_share'])*point+p['win_share']*win,rtol=0,atol=1e-14)
            assert p['actions'][:len(seed_pool)]==[list(x)for x in seed_pool]
            present=set(seed_pool)
            for event in p['events']:
                x=tuple(event['allocation']);assert event['added']==(x not in present)
                if event['added']:present.add(x);admitted+=1
                if event['source']=='large_transfer':
                    parent=event['parent'];transfer=event['transfer'];d,j,n=transfer['donor'],transfer['receiver'],transfer['amount']
                    expected=list(parent);expected[d]-=n;expected[j]+=n
                    assert expected==event['allocation']and d!=j and cfg['large_transfer_min']<=n<=cfg['large_transfer_max']and min(expected)>=0;mutants+=1
                else:
                    pop=Population.from_dict(event['target_population']);br=forward_response(pop.table())
                    assert list(br.allocation)==event['allocation']==event['response']['allocation']and br.expected_score==rational(event['response']['expected_score']);events+=1
            assert present=={tuple(x)for x in p['actions']}
            for t in p['trace']:assert t['fixed_win_share']==p['win_share']and abs(t['official_population_mean']-7.5)<1e-10
            state=read(folder/'checkpoint.json');content=state.pop('state_sha256');assert hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()==content
            assert state['step']==cfg['learning_steps']and state['input_digest']==p['input_digest']
            for snapshot,shares in [('final',final),('average',average)]:
                population=Population.from_dict(p[snapshot]['population']);assert population.entries==population_from_shares(p['actions'],shares,cfg['quantization_mass']).entries
                equal_table(all_tables[name+'_'+snapshot],population.table());certificate=p[snapshot]['certificate'];br=forward_response(population.table())
                assert Fraction(15)-br.expected_score==rational(certificate['security'])and br.expected_score-Fraction(15,2)==rational(certificate['exploitability'])
                learner_tables.append(population.table());learner_records.append(certificate['best_response'])
            print('Reproduced fresh arena: '+name,flush=True)
    objectives=selections['objectives'];optima=[forward_response(t).expected_score for t in st];scale=10**9
    upper=np.array([[[(n*scale+t.denominator-1)//t.denominator for n in row]for row in t.numerators]for t in st],dtype=np.int64)
    lower_opt=np.array([v.numerator*scale//v.denominator for v in optima],dtype=np.int64)
    counts={int(c):n for c,n in front['class_counts'].items()};tie_counts={o['name']:{c:0 for c in counts}for o in objectives};winners={o['name']:{c:None for c in counts}for o in objectives}
    thresholds={}
    for o in objectives:
        thresholds[o['name']]=np.full(31,2**60,dtype=np.int64)
        for c,p in front['frontiers'][o['name']].items():
            v=rational(p['nominal_value']);thresholds[o['name']][int(c)]=-((-v.numerator*scale)//v.denominator)
    lr=np.array([t.numerators for t in learner_tables],dtype=np.int64);lb=[None]*len(lr);lw=[None]*len(lr);lt=[0]*len(lr)
    with np.load(out/'selection-security.npz')as a:saved=a['security_twice']
    iterator=iter_allocations();checked=0;actual_counts={c:0 for c in counts}
    while True:
        batch=list(islice(iterator,cfg['verification_batch_size']))
        if not batch:break
        actions=np.asarray(batch,dtype=np.uint8);security=pattern_security_twice(actions);np.testing.assert_array_equal(security,saved[checked:checked+len(batch)])
        for c in np.unique(security):actual_counts[int(c)]+=int((security==c).sum())
        su=np.zeros((len(st),len(batch)),dtype=np.int64);ls=np.zeros((len(lr),len(batch)),dtype=np.int64)
        for i in range(5):su+=upper[:,i,actions[:,i]];ls+=lr[:,i,actions[:,i]]
        for o in objectives:
            indices=o['indices'];values=su[indices]
            if o['mode']=='regret':values=values-lower_opt[indices,None]
            possible=np.flatnonzero(values.min(axis=0)>=thresholds[o['name']][security])
            for index in possible:
                x=tuple(map(int,actions[index]));c=int(security[index]);vals=[st[i].expected_score(x)for i in indices]
                if o['mode']=='regret':vals=[v-optima[i]for v,i in zip(vals,indices)]
                v=min(vals);target=rational(front['frontiers'][o['name']][str(c)]['nominal_value']);assert v<=target
                if v==target:
                    tie_counts[o['name']][c]+=1;oldw=winners[o['name']][c];winners[o['name']][c]=x if oldw is None else min(oldw,x)
        for j,row in enumerate(ls):
            maximum=int(row.max());positions=np.flatnonzero(row==maximum)
            if lb[j]is None or maximum>lb[j]:lb[j]=maximum;lw[j]=tuple(map(int,actions[positions[0]]));lt[j]=0
            if maximum==lb[j]:lt[j]+=len(positions)
        checked+=len(batch)
        if checked%1000000<cfg['verification_batch_size']or checked==DEFAULT_GAME.actions:print(f'Independent action scan {checked:,}/{DEFAULT_GAME.actions:,}',flush=True)
    assert actual_counts==counts
    for o in objectives:
        for c,p in front['frontiers'][o['name']].items():assert list(winners[o['name']][int(c)])==p['allocation']and tie_counts[o['name']][int(c)]==p['optimum_count']
        allowed=[p for c,p in front['frontiers'][o['name']].items()if int(c)>=4];best=max(rational(p['nominal_value'])for p in allowed);winning=[p for p in allowed if rational(p['nominal_value'])==best]
        p=r['primary'][o['name']];assert p['allocation']==min(v['allocation']for v in winning)and rational(p['selection_value'])==best and p['optimum_count']==sum(v['optimum_count']for v in winning)
    for j,t in enumerate(learner_tables):assert list(lw[j])==learner_records[j]['allocation']and Fraction(lb[j],t.denominator)==rational(learner_records[j]['expected_score'])and lt[j]==learner_records[j]['optimum_count']
    cells=0
    for cid,c in r['candidates'].items():
        x=c['allocation'];assert score(x,r['security'][cid]['counter'])==rational(r['security'][cid]['guarantee'])
        for k,t in all_tables.items():
            value=t.expected_score(x);assert value==rational(r['scores'][cid][k])and rational(r['models'][k]['best_response']['expected_score'])-value==rational(r['regrets'][cid][k]);cells+=1
        for phase in cfg['phases']:
            keys=[k for k,model in r['models'].items()if model['phase']==phase and model['kind']in ['fresh_static','fresh_learning']]
            family=[rational(r['scores'][cid][k])for k in r['families'][phase].values()];s=r['summary'][cid][phase]
            assert len(family)==8 and sum(family)/8==rational(s['family_balanced_mean'])and min(family)==rational(s['family_minimum'])
            minimum=min(rational(r['scores'][cid][k])for k in keys);regret=max(rational(r['regrets'][cid][k])for k in keys)
            assert minimum==rational(s['individual_minimum'])and regret==rational(s['individual_worst_regret'])
            assert s['minimum_scenarios']==[k for k in keys if rational(r['scores'][cid][k])==minimum]and s['worst_regret_scenarios']==[k for k in keys if rational(r['regrets'][cid][k])==regret]
    assert abs(r['sampling_bound']-15*np.sqrt(np.log(10*52/cfg['sampling_alpha'])/(2*cfg['evaluation_sample_size'])))<1e-14
    for name in ['game1-machine-checkpoint-v1.json','game1-merged-checkpoint-v2.json']:
        for path,digest in read(ROOT/'results'/name)['source_artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    a={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
       'frozen_candidates':len(r['candidates']),'frozen_primary_rules':len(r['primary']),'direct_populations_regenerated':sampled,'legal_fresh_draws_checked':sample_rows,
       'arena_paths_reproduced':len(r['learning']),'maximum_replay_absolute_error':maximum_replay_error,'exact_innovation_targets_checked':events,'large_transfer_mutations_checked':mutants,
       'innovations_admitted_across_paths':admitted,'legal_actions_independently_scanned':checked,'conditional_optima_and_tie_counts_checked':len(objectives)*len(counts),
       'new_primary_global_selections_checked':len(objectives),'arena_certificates_independently_enumerated':len(learner_tables),'scenario_tables':len(all_tables),'score_regret_cells_checked':cells,
       'earlier_checkpoints_unchanged':True,'freeze_sha256':r['freeze_sha256'],'scope':'Fresh constructed challenges under predeclared rules; no applicant-frequency inference or final submission'}
    save(out/'acceptance.json',a);print(json.dumps(a,indent=2))


if __name__=='__main__':main()
