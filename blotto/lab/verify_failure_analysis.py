"""Independent forward-DP and full-action security audit for failure diagnostics."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
import argparse
import json
import time
import numpy as np
from blotto_lab.core import BestResponse,DEFAULT_GAME,iter_allocations,mix_tables,score
from blotto_lab.contamination import pattern_security_twice
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha
from verify_population_study import rational


def forward_response(table,first_cap=None):
    """Forward budget DP with explicit reachability; no forbidden-bid penalty."""
    states={0:(0,1,())};evaluations=0
    for i,row in enumerate(table.numerators):
        nxt={}
        for spent,(value,count,prefix)in states.items():
            maximum=table.game.budget-spent
            if i==0 and first_cap is not None:maximum=min(maximum,first_cap)
            for bid in range(maximum+1):
                evaluations+=1;total=spent+bid;candidate=value+row[bid];x=prefix+(bid,)
                old=nxt.get(total)
                if old is None or candidate>old[0]:nxt[total]=(candidate,count,x)
                elif candidate==old[0]:nxt[total]=(candidate,old[1]+count,min(old[2],x))
        states=nxt
    value,count,x=states[table.game.budget]
    return BestResponse(x,Fraction(value,table.denominator),count,evaluations,'independent forward DP')


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/failure-analysis')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');manifest=read(out/'manifest.json')
    for name,digest in manifest['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in manifest['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in manifest['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    prior=read(ROOT/'results/merged-game1/results.json');defs=read(ROOT/'results/merged-game1/development-score-tables.json')
    base=unpack_table({'table':r['baseline']['table']},'baseline')
    expected_base=unpack_table(defs['tables'][defs['search_keys'][defs['midpoint']['table_index']]],'saved midpoint')
    assert base.numerators==expected_base.numerators and base.denominator==expected_base.denominator
    focus={name:p['candidate_id']for name,p in prior['primary'].items()};focus['previous_machine_max_min']=prior['label_ids']['M_known_model_max_min'];assert focus==r['focus']
    selected={};all_keys=[k for k,m in prior['models'].items()if m['phase']!='control']
    for name,cid in focus.items():
        assert r['allocations'][name]==prior['candidates'][cid]['allocation']
        for label,field,keys,mode in [('all_minimum','scores',all_keys,min),('all_regret','regrets',all_keys,max),
             ('audit_regret','regrets',[k for k in all_keys if prior['models'][k]['phase']=='audit'],max)]:
            vals={k:rational(prior[field][cid][k])for k in keys};target=mode(vals.values())
            winners=[k for k in keys if vals[k]==target]
            assert winners==r['extremes'][name][label]['scenarios']and target==rational(r['extremes'][name][label]['value'])
            for k in winners:selected.setdefault(k,[]).append({'candidate':name,'criterion':label})
    assert set(selected)==set(r['critical_scenarios'])
    attribution_cells=segment_endpoints=interiors=boundaries=thresholds=finite_checks=0
    for key,case in r['critical_scenarios'].items():
        assert case['reasons']==selected[key]
        table=unpack_table(prior['models'][key],key);br=forward_response(table)
        assert list(br.allocation)==case['best_response']['allocation']and br.expected_score==rational(case['best_response']['expected_score'])
        atoms=case['marginal_atoms']
        for i,row in enumerate(atoms):
            probs={p['bid']:rational(p['probability'])for p in row}
            assert all(v>0 for v in probs.values())and sum(probs.values())==1
            for bid,number in enumerate(table.numerators[i]):
                expected=table.game.values[i]*(sum(p for k,p in probs.items()if k<bid)+probs.get(bid,0)/Fraction(2))
                assert expected==Fraction(number,table.denominator)
        for name,x in r['allocations'].items():
            data=r['attributions'][name][key]
            assert rational(data['score'])==table.expected_score(x)==rational(prior['scores'][focus[name]][key])
            assert rational(data['regret'])==br.expected_score-table.expected_score(x)==rational(prior['regrets'][focus[name]][key])
            assert list(br.allocation)==data['best_response']
            for i,c in enumerate(data['castles']):
                prob={p['bid']:rational(p['probability'])for p in atoms[i]};win=sum(v for k,v in prob.items()if k<x[i]);tie=prob.get(x[i],0)
                assert rational(c['win'])==win and rational(c['tie'])==tie and rational(c['lose'])==1-win-tie
                actual=Fraction(table.numerators[i][x[i]],table.denominator);counter=Fraction(table.numerators[i][br.allocation[i]],table.denominator)
                assert rational(c['score'])==actual and rational(c['response_score'])==counter and rational(c['regret_contribution'])==counter-actual
            assert sum(rational(c['regret_contribution'])for c in data['castles'])==rational(data['regret'])
            moves=[]
            for i in range(5):
                if not x[i]:continue
                for j in range(5):
                    if i==j:continue
                    y=list(x);y[i]-=1;y[j]+=1;moves.append((table.expected_score(y)-table.expected_score(x),i+1,j+1,y))
            best=max(v[0]for v in moves);assert best==rational(data['best_single_troop_gain'])
            actual=[(rational(v['gain']),v['from_castle'],v['to_castle'],v['allocation'])for v in data['best_single_troop_moves']]
            assert actual==[v for v in moves if v[0]==best];attribution_cells+=1
        for label,cap in [('unrestricted',None),('security2',3)]:
            mapped=r['maps'][key][label];cache={}
            def oracle(t):
                if t not in cache:cache[t]=forward_response(mix_tables([base,table],[1-t,t]),cap)
                return cache[t]
            previous=Fraction(0)
            for seg in mapped['segments']:
                lo,hi=rational(seg['lower_share']),rational(seg['upper_share']);x=seg['allocation'];assert lo==previous<hi
                assert cap is None or x[0]<=cap
                av,bv=base.expected_score(x),table.expected_score(x)
                assert av==rational(seg['baseline_score'])and bv==rational(seg['scenario_score'])
                for t in [lo,hi]:assert (1-t)*av+t*bv==oracle(t).expected_score;segment_endpoints+=1
                middle=oracle((lo+hi)/2);assert list(middle.allocation)==x and middle.optimum_count==seg['open_interval_optimum_count'];interiors+=1;previous=hi
            assert previous==1
            for point in mapped['boundaries']:
                t=rational(point['share']);ordinary=oracle(t)
                assert point['direction']==0 and list(ordinary.allocation)==point['allocation']and ordinary.optimum_count==point['optimum_count']
                assert ordinary.expected_score==rational(point['score']);boundaries+=1
            initial=oracle(Fraction(0)).allocation;assert list(initial)==mapped['baseline_allocation']
            first=mapped['segments'][0];same=base.expected_score(initial)==rational(first['baseline_score'])and table.expected_score(initial)==rational(first['scenario_score'])
            change=rational(first['upper_share'])if same and len(mapped['segments'])>1 else(None if same else Fraction(0))
            assert change==(None if mapped['first_line_change']is None else rational(mapped['first_line_change']))
            # Independently solve each piece for the earliest threshold crossing.
            for raw,saved in mapped['loss_thresholds'].items():
                target=Fraction(raw);earliest=None;av=base.expected_score(initial);bv=table.expected_score(initial)
                for seg in mapped['segments']:
                    lo,hi=rational(seg['lower_share']),rational(seg['upper_share'])
                    v0=rational(seg['baseline_score'])-av;slope=rational(seg['scenario_score'])-rational(seg['baseline_score'])-bv+av
                    if v0+slope*lo>=target:earliest=lo;break
                    if v0+slope*hi>=target:earliest=(target-v0)/slope;break
                assert earliest==(None if saved is None else rational(saved));thresholds+=1
        finite=r['maps'][key]['frozen_candidates']
        def leaders(t):
            values={n:(1-t)*base.expected_score(x)+t*table.expected_score(x)for n,x in r['allocations'].items()};best=max(values.values())
            return sorted(n for n,v in values.items()if v==best),values
        previous=Fraction(0)
        for seg in finite['segments']:
            lo,hi=rational(seg['lower_share']),rational(seg['upper_share']);assert lo==previous<hi
            assert leaders((lo+hi)/2)[0]==seg['winners']
            for t in [lo,hi]:assert set(seg['winners'])<=set(leaders(t)[0]);finite_checks+=1
            previous=hi
        assert previous==1
        for p in finite['boundaries']:assert leaders(rational(p['share']))[0]==p['winners'];finite_checks+=1
        print(f'Certified failure map: {case["short_name"]} {key}',flush=True)
    iterator=iter_allocations();checked=0;cap_count=0
    with np.load(ROOT/'results/merged-game1/security-classes.npz')as a:saved=a['security_twice']
    while True:
        batch=list(islice(iterator,50000))
        if not batch:break
        actions=np.asarray(batch,dtype=np.uint8);security=pattern_security_twice(actions)
        np.testing.assert_array_equal(security,saved[checked:checked+len(batch)])
        np.testing.assert_array_equal(security>=4,actions[:,0]<=3)
        cap_count+=int((actions[:,0]<=3).sum());checked+=len(batch)
    # Old versioned checkpoints stay hash-valid after adding this diagnostic.
    for checkpoint in ['game1-merged-checkpoint-v2.json','game1-machine-checkpoint-v1.json']:
        data=read(ROOT/'results'/checkpoint)
        for name,digest in data['source_artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    result={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
            'critical_scenarios':len(selected),'frozen_focus_candidates':len(focus),'candidate_scenario_attributions':attribution_cells,
            'complete_response_maps':2*len(selected),'global_segment_endpoint_checks':segment_endpoints,'interior_tie_count_checks':interiors,
            'boundary_tie_count_checks':boundaries,'loss_thresholds_checked':thresholds,'finite_envelope_checks':finite_checks,
            'legal_actions_security_equivalence_checked':checked,'allocations_with_security_at_least_two':cap_count,
            'earlier_checkpoints_unchanged':True,'scope':'Conditional known-evidence diagnosis, no new holdout or applicant-frequency inference'}
    save(out/'acceptance.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
