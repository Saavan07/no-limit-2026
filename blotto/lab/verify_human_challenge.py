"""Independent scoring, support, range-selection, and learning audit."""
from fractions import Fraction
from math import comb,factorial
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from blotto_lab.core import Population,best_response,score,score_twice,security_of
from blotto_lab.human import learning_path,ranged_allocations,score_matrices
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha
from verify_population_study import rational


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/human-challenge')
    out=parser.parse_args().study.resolve();clock=time.perf_counter()
    r,m=read(out/'results.json'),read(out/'manifest.json');cfg=r['configuration']
    for relative,digest in m['artifacts'].items():assert sha(out/relative)==digest,relative
    for relative,digest in m['signature']['source_hashes'].items():assert sha(ROOT/relative)==sha(out/'source-snapshot'/relative)==digest,relative
    for relative,digest in m['signature']['input_hashes'].items():assert sha(ROOT/relative)==digest,relative
    baseline=read(ROOT/'results/game1-machine-checkpoint-v1.json')
    for relative,digest in baseline['source_artifact_sha256'].items():assert sha(ROOT/relative)==digest,relative
    assert sha(ROOT/'results/game1-machine-checkpoint-v1.json')==r['baseline_sha256']
    freeze=read(out/'phase1-freeze.json')
    assert sha(out/'phase1-freeze.json')==r['phase1_freeze_sha256']==m['phase1_freeze_sha256']
    assert freeze['created_unix']==r['phase1_frozen_unix']<r['new_models_started_unix']
    assert freeze['labels']==r['labels'] and freeze['pool_actions']==r['pool_actions']
    for label,x in cfg['literal_vectors'].items():assert r['labels'][label]==x
    for label,x in baseline['primary_comparison_vectors'].items():assert r['labels']['M_'+label]==x
    old=read(ROOT/'results/contamination-study/results.json');keys=old['known_model_order']
    tables=[unpack_table(old['known_models'][k],k) for k in keys]
    optima=[best_response(t).expected_score for t in tables]
    ref=unpack_table(old['reference_models']['ref_balanced_s0'],'ref')
    half=unpack_table(old['reference_models']['ref_balanced_s0.50'],'half')
    range_checks=0
    for name,radius in [('literal',0),('relaxed_by_two',cfg['relaxation_radius'])]:
        rows=ranged_allocations(cfg['literal_ranges'],radius)
        saved=r['range_results'][name]
        assert len(rows)==saved['count']==len(saved['records'])
        values={k:[] for k in saved['chosen']}
        for x,record in zip(rows,saved['records']):
            assert list(x)==record['allocation']
            points=[t.expected_score(x) for t in tables]
            actual={'max_min':min(points),'minimax_regret':-max(a-b for a,b in zip(optima,points)),
                'balanced':ref.expected_score(x),'half_strategic':half.expected_score(x)}
            for k,v in actual.items():
                assert v==rational(record[k]);values[k].append((v,x))
            range_checks+=1
        for k,records in values.items():
            maximum=max(v for v,x in records);winners=[x for v,x in records if v==maximum]
            assert rational(saved['chosen'][k]['maximized_value'])==maximum
            assert tuple(saved['chosen'][k]['allocation'])==min(winners)
            assert saved['chosen'][k]['optimum_count']==len(winners)
    for label,x in r['labels'].items():
        summary=r['known_summary'][label];points=[t.expected_score(x) for t in tables]
        assert rational(summary['minimum'])==min(points)
        assert rational(summary['worst_regret'])==max(a-b for a,b in zip(optima,points))
        assert rational(summary['balanced'])==ref.expected_score(x)
        assert rational(summary['half_strategic'])==half.expected_score(x)
        assert score(x,summary['counter'])==rational(summary['security'])
        assert security_of(Population.from_people([tuple(x)]))['security']==rational(summary['security'])
    with (out/'primary-head-to-head.csv').open() as f:
        head=list(csv.reader(f));labels=head[0][1:]
        for row in head[1:]:
            for label,saved in zip(labels,row[1:]):assert float(saved)==float(score(r['labels'][row[0]],r['labels'][label]))
    metrics=read(out/'joint-metrics.json');all_actions=[tuple(x) for x in r['pool_actions']]
    joint_cells=support_records=conditional_checks=0
    for key,model in r['models'].items():
        pop=Population.from_dict(read(out/model['population_file']));table=pop.table()
        assert pop.total_mass==model['total_mass'] and len(pop.entries)==model['support_size']
        assert [list(row) for row in table.numerators]==model['table']['numerators'] and table.denominator==model['table']['denominator']
        target=[i-1 for i in model['coalition']];free=100-model['scout']*(5-len(target))-len(target)
        expected_total=comb(free+len(target)-1,len(target)-1) if model['law']=='uniform' else sum(i+1 for i in target)**free
        if model['heavy5']:
            others=len(target)-1
            if model['law']=='uniform':expected_total=sum(comb(free-k+others-1,others-1) for k in range(cfg['heavy_five_minimum']-1,free+1))
            else:expected_total=sum(comb(free,k)*5**k*sum(i+1 for i in target if i!=4)**(free-k) for k in range(cfg['heavy_five_minimum']-1,free+1))
        assert pop.total_mass==expected_total
        for x,mass in pop.entries:
            assert all(x[i]==model['scout'] for i in range(5) if i not in target)
            assert all(x[i]>=1 for i in target)
            if model['heavy5']:assert x[4]>=cfg['heavy_five_minimum']
            extra=[x[i]-1 for i in target]
            divisor=1
            for a in extra:divisor*=factorial(a)
            exact_mass=1 if model['law']=='uniform' else factorial(free)//divisor
            if model['law']!='uniform':
                for i,a in zip(target,extra):exact_mass*=(i+1)**a
            assert exact_mass==mass
            support_records+=1
        rows=np.array([x for x,mass in pop.entries],dtype=np.int64);masses=np.array([mass for x,mass in pop.entries],dtype=object)
        primary_set={tuple(x) for x in r['labels'].values()}
        for index,x in enumerate(all_actions):
            cid=f'v{index:03}';saved=metrics[key][cid]
            # Independent direct per-field winner calculation.
            u2=np.array([sum((i+1)*(2 if x[i]>y[i] else 1 if x[i]==y[i] else 0) for i in range(5)) for y in rows],dtype=np.int16)
            win_mass=int(sum(masses[u2>15]));tie_mass=int(sum(masses[u2==15]))
            assert rational(saved['win_probability'])==Fraction(win_mass,pop.total_mass)
            assert rational(saved['tie_probability'])==Fraction(tie_mass,pop.total_mass)
            assert rational(saved['win_tie_adjusted'])==Fraction(2*win_mass+tie_mass,2*pop.total_mass)
            assert rational(saved['points'])==table.expected_score(x)
            if x in primary_set:
                assert Fraction(sum(int(s)*a for s,a in zip(u2,masses)),2*pop.total_mass)==rational(saved['joint_mean_points'])
                count234=sum((x[i]>rows[:,i]).astype(np.int8) for i in (1,2,3))
                for event,mask in [('lose5',rows[:,4]>x[4]),('lose5_by20',rows[:,4]>=x[4]+20)]:
                    assert rational(saved[event+'_probability'])==Fraction(int(sum(masses[mask])),pop.total_mass)
                    assert rational(saved[event+'_points_mass'])==Fraction(sum(int(s)*a for s,a in zip(u2[mask],masses[mask])),2*pop.total_mass)
                    assert rational(saved[event+'_win2plus234_mass'])==Fraction(int(sum(masses[mask & (count234>=2)])),pop.total_mass)
                    conditional_checks+=1
            joint_cells+=1
        if len([k for k in r['models'] if k<=key])%8==0:print(f'Verified joint population {key}',flush=True)
    for name,reference in r['references'].items():
        chosen=reference['model_keys'];table=unpack_table(reference,name)
        p5=[Fraction(0) for _ in range(101)]
        for key in chosen:
            population=Population.from_dict(read(out/r['models'][key]['population_file']))
            hist=population.histograms()[4]
            p5=[a+Fraction(b,population.total_mass*len(chosen)) for a,b in zip(p5,hist)]
        for name_bin,lo,hi in [('under25',0,24),('at25',25,25),('26_to40',26,40),('41_to49',41,49),('50_plus',50,100)]:
            assert sum(p5[lo:hi+1])==rational(reference['fifth_bins'][name_bin])
        for index,x in enumerate(all_actions):
            cid=f'v{index:03}';saved=r['reference_scores'][name][cid]
            assert rational(saved['points'])==sum(rational(metrics[k][cid]['points']) for k in chosen)/len(chosen)==table.expected_score(x)
            assert rational(saved['win_tie_adjusted'])==sum(rational(metrics[k][cid]['win_tie_adjusted']) for k in chosen)/len(chosen)
        illustration=r['label_ids']['R_illustration']
        for event in ['lose5','lose5_by20']:
            probability=sum(rational(metrics[k][illustration][event+'_probability']) for k in chosen)/len(chosen)
            mass=sum(rational(metrics[k][illustration][event+'_points_mass']) for k in chosen)/len(chosen)
            assert probability==rational(r['illustration_events'][name][event+'_probability'])
            if probability:assert mass/probability==rational(r['illustration_events'][name][event+'_conditional_points'])
    points2,wins2=score_matrices(all_actions)
    for i in range(0,len(all_actions),17):
        for j in range(0,len(all_actions),19):assert points2[i,j]==score_twice(all_actions[i],all_actions[j])
    learning_certificates=0
    for key,entry in r['learning'].items():
        with np.load(out/'learning'/f'{key}.npz') as arrays:
            initial=arrays['initial'];final=arrays['final'];average=arrays['average']
        fitness=points2/2 if entry['fitness']=='points' else wins2/2
        reproduced=learning_path(fitness,initial,entry['rule'],entry['mutation'],cfg['learning_steps'],cfg['record_every'])
        np.testing.assert_allclose(final,reproduced['final'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(average,reproduced['average'],rtol=0,atol=1e-14)
        expected_mean=7.5 if entry['fitness']=='points' else .5
        for record in entry['trace']:assert abs(record['mean_fitness']-expected_mean)<1e-10
        for phase,array in [('final',final),('average',average)]:
            pop=Population.from_dict(entry[phase]['population']);assert pop.total_mass==cfg['quantization_mass']
            exact=security_of(pop)
            assert exact['exploitability']==rational(entry[phase]['certificate']['exploitability'])
            assert exact['security']==rational(entry[phase]['certificate']['security'])
            mass_lookup=dict(pop.entries);masses=np.array([mass_lookup.get(x,0) for x in all_actions],dtype=np.int64)
            point_gap=Fraction(int((points2.astype(np.int64)@masses).max()),2*pop.total_mass)-Fraction(15,2)
            win_gap=Fraction(int((wins2.astype(np.int64)@masses).max()),2*pop.total_mass)-Fraction(1,2)
            assert point_gap==rational(entry[phase]['restricted_point_exploitability'])
            assert win_gap==rational(entry[phase]['restricted_win_exploitability'])
            assert exact['exploitability']>=point_gap
            assert np.abs(masses/pop.total_mass-array).sum()<len(array)/pop.total_mass*2
            learning_certificates+=1
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
        'verifier_sha256':sha(Path(__file__)),'baseline_unchanged':True,'literal_vectors_verified':5,
        'primary_machine_vectors_unchanged':9,'range_allocations_checked':range_checks,
        'coalition_populations_verified':len(r['models']),'exact_support_records_checked':support_records,
        'joint_candidate_population_cells_checked':joint_cells,'conditional_event_checks':conditional_checks,
        'learning_paths_reproduced':len(r['learning']),'exact_fullspace_point_certificates_checked':learning_certificates,
        'pool_size':len(all_actions),'source_and_freeze_hashes_verified':True,
        'scope':'Exact model scoring and deterministic constructed dynamics; no inference of actual applicant frequencies'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
