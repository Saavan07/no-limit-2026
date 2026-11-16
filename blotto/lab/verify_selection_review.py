"""Independent rank, mass, direct scoring, and selection accounting."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse,json,time
import numpy as np
from blotto_lab.core import Population,score,security_of,validate_allocation
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/selection-review')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');manifest=read(out/'manifest.json')
    for name,digest in manifest['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in manifest['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in manifest['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    old=read(ROOT/'results/final-game1/results.json');freeze=read(ROOT/'results/final-game1/frozen-candidates.json')
    assert old['freeze_sha256']==r['final_freeze_sha256']==sha(ROOT/'results/final-game1/frozen-candidates.json')and old['candidates']==freeze['candidates']
    x=validate_allocation(r['decision']['allocation']);rep=(3,6,25,36,30);assert x==(3,8,4,44,41)
    assert(out/'selected-game1.txt').read_text()==','.join(map(str,x))+'\n'
    assert security_of(Population.from_people([x]))['security']==rational(r['decision']['pure_security'])==2
    for phase in ['challenge','confirmation']:
        eligible=[p for p in old['primary'].values()if rational(old['security'][p['candidate_id']]['guarantee'])>=2]
        assert rational(old['summary'][r['decision']['candidate_id']][phase]['family_balanced_mean'])==max(rational(old['summary'][p['candidate_id']][phase]['family_balanced_mean'])for p in eligible)
    component={};families=defaultdict(Fraction);phases=defaultdict(Fraction)
    for p in r['model_weights']:
        w=rational(p['benchmark_weight']);families[p['family']]+=w;phases[p['phase']]+=w;component[p['model']]=w
    assert sum(component.values())==1 and len(families)==8 and all(v==Fraction(1,8)for v in families.values())and all(v==Fraction(1,2)for v in phases.values())
    tables={k:unpack_table(m,k)for k,m in old['models'].items()};lookup={tuple(c['allocation']):cid for cid,c in old['candidates'].items()}
    metrics={cid:{'benchmark':sum(rational(old['summary'][cid][p]['family_balanced_mean'])for p in ['challenge','confirmation'])/2,
                 'arena_final':sum(rational(old['scores'][cid][n+'_final'])for n in old['learning'])/12,
                 'arena_average':sum(rational(old['scores'][cid][n+'_average'])for n in old['learning'])/12}for cid in old['candidates']}
    assert set(metrics)==set(r['frozen_metrics'])
    for cid,fields in metrics.items():
        assert sum(rational(old['scores'][cid][k])*w for k,w in component.items())==fields['benchmark']
        for field,value in fields.items():
            assert rational(r['frozen_metrics'][cid][field])==value
            vals=[v[field]for v in metrics.values()];rank=r['frozen_metrics'][cid]['ranks'][field]
            assert rank['rank']==1+sum(v>value for v in vals)and rank['ties']==sum(v==value for v in vals)and rank['field_size']==503
    range_count=0
    for label,bounds in [('current',[(3,3),(5,7),(22,25),(35,40),(25,30)]),('original',[(2,3),(5,7),(22,25),(36,42),(25,30)])]:
        rows=[a for a in product(*(range(lo,hi+1)for lo,hi in bounds))if sum(a)==100];records=r['ranges'][label]['records']
        assert {tuple(v['allocation'])for v in records}==set(rows)and len(records)==(42 if label=='current'else 114)
        actual=[]
        for record in records:
            a=tuple(record['allocation']);phase_scores={p:sum(tables[k].expected_score(a)for k in old['families'][p].values())/8 for p in ['challenge','confirmation']}
            vals={'benchmark':sum(phase_scores.values())/2,'arena_final':sum(tables[n+'_final'].expected_score(a)for n in old['learning'])/12,
                  'arena_average':sum(tables[n+'_average'].expected_score(a)for n in old['learning'])/12}
            existing=a in lookup;assert record['candidate_id']==lookup.get(a)
            for field,value in vals.items():
                assert rational(record[field])==value;field_values=[v[field]for v in metrics.values()];rank=record['ranks'][field]
                assert rank['rank']==1+sum(v>value for v in field_values)and rank['field_size']==503+(not existing)and rank['ties']==sum(v==value for v in field_values)+(not existing)
            actual.append((vals['benchmark'],a));range_count+=1
        assert tuple(r['ranges'][label]['best']['allocation'])==min(a for v,a in actual if v==max(v for v,a in actual))
        assert r['ranges'][label]['frozen_individual_count']==sum(a in lookup for a in rows)
    pooled={'final':defaultdict(Fraction),'average':defaultdict(Fraction)};arena_rows=0;direct_scores=0
    saved_snapshots={(p['arena'],p['snapshot']):p for p in r['arena_snapshots']}
    for name,p in old['learning'].items():
        for snapshot in ['final','average']:
            pop=Population.from_dict(p[snapshot]['population']);saved=saved_snapshots[name,snapshot]
            assert saved['total_mass']==pop.total_mass and saved['support']==len(pop.entries)
            records={tuple(v['allocation']):v for v in saved['allocations']};assert set(records)=={a for a,m in pop.entries}
            for a,m in pop.entries:
                v=records[a];share=Fraction(m,pop.total_mass)
                assert v['mass']==m and rational(v['share'])==share and rational(v['percentage'])==100*share
                pooled[snapshot][a]+=share/12;arena_rows+=1
            assert sum(rational(v['share'])for v in saved['allocations'])==1
            for a,field in [(x,'choice_score'),(rep,'human_representative_score')]:
                exact=sum(m*score(a,b)for b,m in pop.entries)/pop.total_mass;assert exact==rational(saved[field]);direct_scores+=1
    for snapshot in ['final','average']:
        saved=r['pooled_'+snapshot];assert sum(rational(v['share'])for v in saved)==1 and len(saved)==len(pooled[snapshot])
        for v in saved:assert rational(v['share'])==pooled[snapshot][tuple(v['allocation'])]
        assert saved==sorted(saved,key=lambda v:(-rational(v['share']),v['allocation']))
    for key,m in old['models'].items():
        if m['kind']!='fresh_static':continue
        with np.load(ROOT/'results/final-game1'/m['sample_file'])as a:rows=a['allocations']
        for allocation in [x,rep]:
            twice=np.zeros(len(rows),dtype=np.int16)
            for j,value in enumerate(range(1,6)):twice+=2*value*(allocation[j]>rows[:,j])+value*(allocation[j]==rows[:,j])
            assert Fraction(int(twice.sum()),2*len(rows))==tables[key].expected_score(allocation);direct_scores+=1
    result={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
        'frozen_candidate_rank_rows_checked':len(metrics),'range_vector_score_and_rank_rows_checked':range_count,'arena_allocation_share_rows_checked':arena_rows,
        'joint_score_crosschecks':direct_scores,'pooled_final_allocations':len(pooled['final']),'pooled_average_allocations':len(pooled['average']),
        'weight_sums_verified':True,'recommendation':x,'final_freeze_unchanged':True,
        'scope':'Review recommendation, explicit hypothetical weights, retrospective range accounting; no applicant forecast'}
    save(out/'acceptance.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
