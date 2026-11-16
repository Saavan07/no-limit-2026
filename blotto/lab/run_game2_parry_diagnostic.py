"""Retrospective exact-vector comparison on the unchanged frozen final bank."""
from fractions import Fraction
from pathlib import Path
from itertools import combinations_with_replacement
from datetime import datetime,timezone
import json,time
import numpy as np
from blotto_lab.core import validate_allocation
from game2_lab.core import focal_samples,reference_group
from game2_lab.human import payoff_arrays
from game2_lab.weight_counters import worst_clone_population
from run_population_study import ROOT,save,sha,write_csv
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def main():
    out=ROOT/'results/game2-parry-initial-comparison'
    if out.exists():raise SystemExit('Diagnostic already exists; preserve it.')
    study=ROOT/'results/game2-final';r=read(study/'results.json');cfg=r['configuration'];N=cfg['evaluation_triplets'];freeze=study/'candidate-freeze.json';frozen_hash=sha(freeze);record=ROOT/'results/game2-selected-v1.json';record_hash=sha(record)
    actions={'parry':(0,36,0,64,0),'current_recommendation':tuple(r['selected_allocation']),'previous_two_four':(0,30,0,68,2),'previous_compromise':(0,1,10,86,3)}
    for x in actions.values():validate_allocation(x)
    weights={g:{k:rational(v)for k,v in w.items()}for g,w in r['policy']['block_weights'].items()};data={name:{}for name in actions};castle={name:{}for name in actions};paired={g:{}for g in weights};direct_cells=0;clock=time.perf_counter()
    for key in r['cell_keys']:
        with np.load(study/'evaluation'/(key+'.npz'))as f:rows=f['triplets']
        quick,quickcastle=payoff_arrays(list(actions.values()),rows)
        for j,(name,x)in enumerate(actions.items()):
            direct=focal_samples(x,rows);np.testing.assert_array_equal(quick[j],direct['payout4']);np.testing.assert_array_equal(quickcastle[j],direct['castle12'])
            data[name][key]=Fraction(int(direct['payout4'].sum()),4*N);castle[name][key]=Fraction(int(direct['castle12'].sum()),12*N);direct_cells+=len(rows)
    summary={}
    for name in actions:
        blocks={g:sum(mass*data[name][key]for key,mass in w.items())for g,w in weights.items()};raw={g:sum(mass*castle[name][key]for key,mass in w.items())for g,w in weights.items()}
        gaps={g:rational(r['policy']['block_benchmarks'][g])-score for g,score in blocks.items()}
        summary[name]={'block_scores':blocks,'block_castle_points':raw,'equal_block_mean':sum(blocks.values())/3,'regret_to_original_frozen_benchmarks':gaps,'largest_regret_to_original_frozen_benchmarks':max(gaps.values())}
    selected=r['policy']['selected_candidate_id']
    for g,score in summary['current_recommendation']['block_scores'].items():assert score==rational(r['policy']['block_scores'][selected][g])
    # Descriptive placement against the old 191-vector ranking, not an amended final rank.
    risk=summary['parry']['largest_regret_to_original_frozen_benchmarks'];mean=summary['parry']['equal_block_mean'];sortkey=lambda cid:(rational(r['policy']['maximum_block_regret'][cid]),-rational(r['policy']['equal_block_mean'][cid]),r['candidates'][cid]['allocation'])
    would_precede=sum(sortkey(cid)<(risk,-mean,list(actions['parry']))for cid in r['candidates'])
    mixtures=[]
    for other_copies_of_current in range(4):
        opponents=[actions['current_recommendation']]*other_copies_of_current+[actions['parry']]*(3-other_copies_of_current)
        mixtures.append({'other_current_copies':other_copies_of_current,'current_focal':reference_group([actions['current_recommendation']]+opponents),'parry_focal':reference_group([actions['parry']]+opponents)})
    # Concrete counterexample to drawing conclusions from two-player comparison alone.
    group_cases=[]
    for n_current in range(5):
        group=[actions['current_recommendation']]*n_current+[actions['parry']]*(4-n_current)
        group_cases.append({'current_copies':n_current,'group':group,'outcome':reference_group(group)})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'actions':actions,'summary':summary,'scores':data,'castle_scores':castle,'two_strategy_group_cases':group_cases,
       'paired_difference_current_minus_parry':{g:summary['current_recommendation']['block_scores'][g]-summary['parry']['block_scores'][g]for g in weights},
       'original_frozen_candidates_preceding_parry_under_fixed_benchmarks':would_precede,'known_comparison_scope':'Descriptive insertion comparison only; no original rank or benchmark is changed',
       'exact_pure_population_counters':{name:worst_clone_population(x)for name,x in actions.items()},'parry_exact_vector_in_original_freeze':False,
       'foundation_cells_checked':direct_cells,'elapsed_seconds':time.perf_counter()-clock,'freeze_sha256':frozen_hash,'selected_record_sha256':record_hash,
       'input_sha256':{'results/game2-final/results.json':sha(study/'results.json'),'results/game2-final/manifest.json':sha(study/'manifest.json')},
       'source_sha256':sha(Path(__file__)),'scope':'Post-freeze diagnostic on our existing models; not an audit of Parry arena, a fresh holdout, or an applicant forecast'}
    out.mkdir();save(out/'results.json',result);write_csv(out/'block-comparison.csv',['name','allocation','broad_mean','pressure_mean','arena_mean','equal_block_mean','largest_regret_to_original_benchmarks'],[[name,str(actions[name]),*[float(summary[name]['block_scores'][g])for g in ['core','pressure','arenas']],float(summary[name]['equal_block_mean']),float(summary[name]['largest_regret_to_original_frozen_benchmarks'])]for name in actions])
    assert sha(freeze)==frozen_hash and sha(record)==record_hash
    save(out/'acceptance.json',{'passed':True,'all_four_vectors_crosschecked_between_scorers':True,'foundation_candidate_group_cells':direct_cells,'original_freeze_and_selection_record_unchanged':True,'source_sha256':sha(Path(__file__)),'results_sha256':sha(out/'results.json')})
    print(json.dumps({'summary':summary,'original_candidates_preceding_parry':would_precede,'scope':result['scope']},indent=2,default=str))


if __name__=='__main__':main()
