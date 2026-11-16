"""Exact selection review, honest population accounting, and human-range ranks."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import shutil
import time
from blotto_lab.core import Population,mix_tables,score,security_of,validate_allocation
from blotto_lab.human import ranged_allocations
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha,write_csv
from verify_population_study import rational


CURRENT=((3,3),(5,7),(22,25),(35,40),(25,30))
ORIGINAL=((2,3),(5,7),(22,25),(36,42),(25,30))
FAMILY_LABELS={'soft':'Soft coalition priorities','heaped':'Round-number priorities with noise',
 'middle5':'Middle-range castle-5 bids','contrarian':'Contrarian coalition switching',
 'four_field':'Four-field priorities','diffuse':'Diffuse all-field allocations',
 'arena_final':'Final arena snapshots','arena_average':'Time-average arena snapshots'}


def read(p):return json.loads(p.read_text())


def in_region(x,bounds):return all(a<=v<=b for v,(a,b)in zip(x,bounds))


def ranking(value,values,existing):
    return {'rank':1+sum(v>value for v in values),'ties':sum(v==value for v in values)+(0 if existing else 1),
            'field_size':len(values)+(0 if existing else 1),'frozen_individual':existing}


def pooled(learning,snapshot):
    mass=defaultdict(Fraction)
    for p in learning.values():
        population=Population.from_dict(p[snapshot]['population'])
        for x,n in population.entries:mass[x]+=Fraction(n,population.total_mass*len(learning))
    assert sum(mass.values())==1
    return [{'allocation':x,'share':v,'percentage':100*v}for x,v in sorted(mass.items(),key=lambda p:(-p[1],p[0]))]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/selection-review')
    out=parser.parse_args().output.resolve()
    if out.exists():raise SystemExit('Output exists. Choose a new review folder.')
    clock=time.perf_counter();out.mkdir(parents=True)
    sources=[Path(__file__),ROOT/'verify_selection_review.py',ROOT/'SELECTION-REVIEW-PROTOCOL.md',ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/human.py',ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py',ROOT/'verify_population_study.py']
    inputs=[ROOT/'results/final-game1/results.json',ROOT/'results/final-game1/acceptance.json',ROOT/'results/final-game1/frozen-candidates.json',
            ROOT/'results/merged-game1/results.json',ROOT/'results/human-challenge/results.json',ROOT/'results/human-challenge/phase1-freeze.json',
            ROOT/'results/game1-final-review-checkpoint-v1.json']
    signature={'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs}}
    save(out/'run-inputs.json',signature)
    for p in sources:
        target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    r=read(inputs[0]);old=read(inputs[3]);human=read(inputs[4]);tables={k:unpack_table(m,k)for k,m in r['models'].items()}
    actions={cid:tuple(c['allocation'])for cid,c in r['candidates'].items()};lookup={x:cid for cid,x in actions.items()}
    family_keys=[k for phase in ['challenge','confirmation']for k in r['families'][phase].values()]
    benchmark=mix_tables([tables[k]for k in family_keys],[1]*16)
    final_tables=[Population.from_dict(p['final']['population']).table()for p in r['learning'].values()]
    average_tables=[Population.from_dict(p['average']['population']).table()for p in r['learning'].values()]
    arena=mix_tables(final_tables,[1]*12);arena_average=mix_tables(average_tables,[1]*12)
    metrics={cid:{'benchmark':benchmark.expected_score(x),'arena_final':arena.expected_score(x),'arena_average':arena_average.expected_score(x)}for cid,x in actions.items()}
    values={m:[v[m]for v in metrics.values()]for m in ['benchmark','arena_final','arena_average']}
    for cid,m in metrics.items():
        m['ranks']={key:ranking(m[key],values[key],True)for key in values}
        m['allocation']=actions[cid];m['primary_roles']=[n for n,p in r['primary'].items()if p['candidate_id']==cid]
        m['pure_security']=rational(r['security'][cid]['guarantee'])
    eligible={name:p for name,p in r['primary'].items()if rational(r['security'][p['candidate_id']]['guarantee'])>=2}
    leaders={phase:[name for name,p in eligible.items()if rational(r['summary'][p['candidate_id']][phase]['family_balanced_mean'])==max(rational(r['summary'][q['candidate_id']][phase]['family_balanced_mean'])for q in eligible.values())]for phase in ['challenge','confirmation']}
    common=set(leaders['challenge'])&set(leaders['confirmation']);assert len(common)==1
    chosen_name=next(iter(common));chosen=r['primary'][chosen_name];chosen_id=chosen['candidate_id'];x=actions[chosen_id]
    assert x==(3,8,4,44,41);validate_allocation(x)
    # Review all frozen allocations without silently promoting exploratory winners.
    all_leaders=sorted(metrics,key=lambda cid:(-metrics[cid]['benchmark'],actions[cid]))[:15]
    skeys=[k for phase in ['challenge','confirmation']for g,k in r['families'][phase].items()if not g.startswith('arena_')]
    lkeys=[k for phase in ['challenge','confirmation']for g,k in r['families'][phase].items()if g.startswith('arena_')]
    static=mix_tables([tables[k]for k in skeys],[1]*12);learned=mix_tables([tables[k]for k in lkeys],[1]*4)
    comparisons={}
    for cid in all_leaders:
        y=actions[cid];ds=static.expected_score(y)-static.expected_score(x);dl=learned.expected_score(y)-learned.expected_score(x)
        crossing=ds/(ds-dl)if ds!=dl else None
        comparisons[cid]={'allocation':y,'static_score':static.expected_score(y),'learned_score':learned.expected_score(y),
                         'difference_from_choice':metrics[cid]['benchmark']-metrics[chosen_id]['benchmark'],
                         'learner_share_crossing':crossing if crossing is not None and 0<=crossing<=1 else None}
    primary={name:{**metrics[p['candidate_id']],'candidate_id':p['candidate_id'],'phase_summary':r['summary'][p['candidate_id']]}for name,p in r['primary'].items()}
    range_data={};range_rows=[]
    for label,bounds in [('current',CURRENT),('original',ORIGINAL)]:
        rows=ranged_allocations(bounds);records=[]
        for y in rows:
            cid=lookup.get(y);bm=benchmark.expected_score(y);av=arena.expected_score(y);aa=arena_average.expected_score(y)
            record={'allocation':y,'candidate_id':cid,'benchmark':bm,'arena_final':av,'arena_average':aa,
                    'ranks':{m:ranking(v,values[m],cid is not None)for m,v in [('benchmark',bm),('arena_final',av),('arena_average',aa)]},
                    'phase_means':{phase:sum(tables[k].expected_score(y)for k in r['families'][phase].values())/8 for phase in ['challenge','confirmation']},
                    'prior_arena_seed':y in {tuple(a)for a in human['pool_actions']}}
            records.append(record);range_rows.append((label,record))
        records.sort(key=lambda v:(-v['benchmark'],v['allocation']))
        range_data[label]={'bounds':bounds,'size':len(rows),'frozen_individual_count':sum(v['candidate_id']is not None for v in records),
                           'initial_arena_seed_count':sum(v['prior_arena_seed']for v in records),
                           'best':records[0],'worst':records[-1],
                           'median_score':(records[(len(records)-1)//2]['benchmark']+records[len(records)//2]['benchmark'])/2,
                           'records':records}
    representative=(3,6,25,36,30);rep_id=lookup[representative]
    prior_fit=human['range_results']['literal']['chosen']
    assert representative in [tuple(p['allocation'])for p in prior_fit.values()]
    snapshots=[];within_rows=[];range_presence=[]
    for name,path in r['learning'].items():
        for snapshot in ['final','average']:
            pop=Population.from_dict(path[snapshot]['population']);total=pop.total_mass;entries=sorted(pop.entries,key=lambda p:(-p[1],p[0]))
            rows=[{'allocation':a,'mass':m,'share':Fraction(m,total),'percentage':Fraction(100*m,total)}for a,m in entries]
            for entry in rows:
                within_rows.append((name,path['phase'],snapshot,path['win_share'],path['mutation'],entry,total))
            snapshots.append({'arena':name,'phase':path['phase'],'snapshot':snapshot,'fixed_win_share':Fraction(str(path['win_share'])),
                              'mutation':path['mutation'],'support':len(entries),'total_mass':total,
                              'choice_score':pop.reference_mean(x),'human_representative_score':pop.reference_mean(representative),
                              'choice_share':Fraction(dict(pop.entries).get(x,0),total),'allocations':rows})
            range_presence.append({'arena':name,'snapshot':snapshot,**{label:{'share':Fraction(sum(m for a,m in entries if in_region(a,bounds)),total),
                           'positive_allocations':sum(in_region(a,bounds)for a,m in entries)}for label,bounds in [('current',CURRENT),('original',ORIGINAL)]}})
    pooled_final=pooled(r['learning'],'final');pooled_average=pooled(r['learning'],'average')
    assert len(pooled_final)==783
    model_weights=[]
    for phase in ['challenge','confirmation']:
        for group,key in r['families'][phase].items():
            family=r['models'][key];parts=family['component_keys'];weights=[Fraction(1,len(parts))]*len(parts)
            for component,w in zip(parts,weights):model_weights.append({'phase':phase,'family':group,'model':component,'benchmark_weight':Fraction(1,16)*w})
    assert sum(p['benchmark_weight']for p in model_weights)==1
    old_id=next(cid for cid,c in old['candidates'].items()if tuple(c['allocation'])==x)
    review={'decision':{'rule':chosen_name,'candidate_id':chosen_id,'allocation':x,'submission_format':','.join(map(str,x)),
                'basis':'Unique security-eligible primary leading the previously declared family-average score in both fresh phases; no new vector fitted here',
                'benchmark':metrics[chosen_id]['benchmark'],'ranks':metrics[chosen_id]['ranks'],'pure_security':metrics[chosen_id]['pure_security'],
                'phase_summary':r['summary'][chosen_id],'earlier_summary':old['summary'][old_id],
                'uncertainty':'No empirical applicant-frequency estimates. Constructed-model weights and learning frequencies are assumptions and outcomes of simulations.'},
            'primary':primary,'frozen_metrics':metrics,'exploratory_leaders':all_leaders,'exploratory_comparisons':comparisons,
            'benchmark_family_weights':{k:Fraction(1,8)for k in FAMILY_LABELS},'model_weights':model_weights,
            'static_mean':static.expected_score(x),'learned_mean':learned.expected_score(x),
            'pooled_final':pooled_final,'pooled_average':pooled_average,'arena_snapshots':snapshots,'range_presence':range_presence,
            'ranges':range_data,'frozen_human_representative':{'candidate_id':rep_id,**metrics[rep_id],'phase_summary':r['summary'][rep_id]},
            'earlier_illustration':{'candidate_id':lookup[(2,6,24,40,28)],**metrics[lookup[(2,6,24,40,28)]]},
            'official_rules_source':'https://pokerbot.sequoiacap.com/','official_scoring_verified':'100 integer troops, values 1..5, ties split, score is average castle points against applicants',
            'final_freeze_sha256':r['freeze_sha256'],'scope':'One-vector Game 1 recommendation and retrospective accounting; no application submitted; Game 2 reserved'}
    save(out/'results.json',review);(out/'selected-game1.txt').write_text(','.join(map(str,x))+'\n')
    write_csv(out/'frozen-rankings.csv',['candidate_id','castle1','castle2','castle3','castle4','castle5','primary_roles','family_score','family_rank','final_arena_score','final_arena_rank','time_average_score','time_average_rank','pure_security'],
       [[cid,*actions[cid],';'.join(v['primary_roles']),float(v['benchmark']),v['ranks']['benchmark']['rank'],float(v['arena_final']),v['ranks']['arena_final']['rank'],float(v['arena_average']),v['ranks']['arena_average']['rank'],float(v['pure_security'])]for cid,v in sorted(metrics.items(),key=lambda p:(-p[1]['benchmark'],p[1]['allocation']))])
    write_csv(out/'human-range-rankings.csv',['range_set','castle1','castle2','castle3','castle4','castle5','frozen_id','prior_arena_seed','family_score','family_rank','family_field_size','final_arena_score','final_arena_rank','final_arena_field_size','challenge_mean','confirmation_mean'],
       [[label,*v['allocation'],v['candidate_id'],v['prior_arena_seed'],float(v['benchmark']),v['ranks']['benchmark']['rank'],v['ranks']['benchmark']['field_size'],float(v['arena_final']),v['ranks']['arena_final']['rank'],v['ranks']['arena_final']['field_size'],float(v['phase_means']['challenge']),float(v['phase_means']['confirmation'])]for label,v in range_rows])
    for label,rows in [('final',pooled_final),('average',pooled_average)]:
        write_csv(out/('pooled-'+label+'-strategy-shares.csv'),['castle1','castle2','castle3','castle4','castle5','percentage','share_numerator','share_denominator'],
                  [[*v['allocation'],float(v['percentage']),v['share'].numerator,v['share'].denominator]for v in rows])
    write_csv(out/'all-arena-strategy-shares.csv',['arena','phase','snapshot','win_type_share','mutation','castle1','castle2','castle3','castle4','castle5','integer_mass','total_mass','within_arena_percentage','benchmark_percentage_contribution'],
       [[name,phase,snapshot,ws,mu,*entry['allocation'],entry['mass'],total,float(entry['percentage']),float(entry['percentage']/96)]for name,phase,snapshot,ws,mu,entry,total in within_rows])
    write_csv(out/'benchmark-component-weights.csv',['phase','family','scenario','benchmark_percentage','weight_numerator','weight_denominator'],
       [[p['phase'],p['family'],p['model'],float(100*p['benchmark_weight']),p['benchmark_weight'].numerator,p['benchmark_weight'].denominator]for p in model_weights])
    write_csv(out/'arena-summary.csv',['arena','snapshot','support','fixed_win_type_percentage','mutation','selected_score','human_representative_score','selected_population_percentage','top_allocation','top_percentage'],
       [[p['arena'],p['snapshot'],p['support'],float(p['fixed_win_share']*100),p['mutation'],float(p['choice_score']),float(p['human_representative_score']),float(p['choice_share']*100),str(p['allocations'][0]['allocation']),float(p['allocations'][0]['percentage'])]for p in snapshots])
    write_csv(out/'range-population-shares.csv',['arena','snapshot','range_set','percentage','positive_allocations'],
       [[p['arena'],p['snapshot'],label,float(p[label]['share']*100),p[label]['positive_allocations']]for p in range_presence for label in ['current','original']])
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'elapsed_seconds':time.perf_counter()-clock,'completed_unix':time.time(),
                            'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Selection review complete:',','.join(map(str,x)),flush=True)


if __name__=='__main__':main()
