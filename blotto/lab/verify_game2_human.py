"""Independent exact and full-primary audit of Game 2 human hypotheses."""
from fractions import Fraction
from math import comb,factorial
from pathlib import Path
import argparse,json,time
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import focal_samples,reference_group,fixed_sample_bound
from game2_lab.human import fresh_triplets,payoff_arrays
from run_multiwave_study import ROOT
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def independent_weights(ids,counts):
    result=[]
    for triple in ids:
        a,b,c=map(int,triple)
        factor=1 if a==c else(3 if a==b or b==c else 6)
        result.append(factor*counts[a]*counts[b]*counts[c])
    assert sum(result)==sum(counts)**3
    return np.array(result,dtype=np.int64)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-human')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');m=read(out/'manifest.json');cfg=r['configuration']
    for name,digest in m['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in m['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in m['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    freeze=read(out/'frozen-candidates.json');start=read(out/'evaluation-start.json');freeze_hash=sha(out/'frozen-candidates.json')
    assert freeze_hash==r['freeze_sha256']==m['freeze_sha256']==start['freeze_sha256']and freeze['created_unix']<start['started_unix']
    for key in ['candidates','primary','shortlist']:assert freeze[key]==r[key]
    for file,key in [('exact-arrays.npz','exact_arrays_sha256'),('grid-definitions.json','grid_definitions_sha256'),('run-inputs.json','run_inputs_sha256')]:assert sha(out/file)==freeze[key]
    d=read(out/'grid-definitions.json');actions=[tuple(v['allocation'])for v in r['candidates'].values()];cids=list(r['candidates'])
    with np.load(out/'exact-arrays.npz')as a:ids=a['anchor_type_ids'];payout=a['payout4'];castle=a['castle12'];weights=a['grid_weights'];scores=a['grid_scores'];centerweights=a['center_weights'];centers=a['center_scores']
    support=np.array(d['support']);triplets=support[ids]
    assert len(d['mixes'])==r['grid_scenarios']==286
    for i,counts in enumerate(d['counts']):np.testing.assert_array_equal(weights[:,i],independent_weights(ids,counts))
    np.testing.assert_array_equal(centerweights,independent_weights(ids,d['center_counts']))
    exact_cells=0
    for i,x in enumerate(actions):
        expected=focal_samples(x,triplets);np.testing.assert_array_equal(payout[i],expected['payout4']);np.testing.assert_array_equal(castle[i],expected['castle12']);exact_cells+=len(ids)
    np.testing.assert_array_equal(scores,payout.astype(np.int64)@weights);np.testing.assert_array_equal(centers,payout.astype(np.int64)@centerweights)
    # The anchor mixture's own symmetric self-play must average 3.75.
    counts=d['center_counts'];indices=[actions.index(tuple(x))for x in d['support']]
    assert Fraction(sum(int(n)*int(centers[i])for n,i in zip(counts,indices)),d['center_denominator']*sum(counts))==Fraction(15,4)
    for name,p in r['primary'].items():
        i=actions.index(tuple(p['allocation']));assert r['candidates'][p['candidate_id']]['allocation']==p['allocation']
        assert rational(r['exact_primary'][name]['center'])==Fraction(int(centers[i]),d['center_denominator'])
        assert rational(r['exact_primary'][name]['worst_grid'])==Fraction(int(scores[i].min()),d['denominator'])
        if name=='zero_five_center':allowed=[j for j,x in enumerate(actions)if x[4]==0]
        elif name=='limp_five_center':allowed=[j for j,x in enumerate(actions)if 1<=x[4]<=3]
        elif name=='broad_center':allowed=[j for j,c in enumerate(r['candidates'].values())if 'broad_seeded'in c['roles']or'anchor_broad'in c['roles']]
        elif name in ['overall_center','worst_grid']:allowed=list(range(len(actions)))
        else:continue
        value=scores.min(axis=1)if name=='worst_grid'else centers;maximum=max(int(value[j])for j in allowed)
        assert int(value[i])==maximum and tuple(p['allocation'])==min(actions[j]for j in allowed if int(value[j])==maximum)
    clone_witnesses=0
    for k,cases in r['clone_cases'].items():
        if k in ['1','2','3']:
            for v,case in cases['outcomes'].items():
                outcome=reference_group(case['group']);assert outcome.payouts[0]==Fraction(int(v),4);clone_witnesses+=1
        else:
            outcome=reference_group(cases['group']);assert list(outcome.payouts)==[rational(v)for v in cases['outcome']['payouts']];clone_witnesses+=1
    curve_points=0
    for profile,data in r['copy_curves'].items():
        a=[rational(v)for v in data['conditional_payouts_by_other_copy_count']];coeff=[rational(v)for v in data['monomial_coefficients']]
        assert a[0]>=4 and a[1]>=1 and a[2]>=2 and a[3]==Fraction(15,4)
        for point in data['curve']:
            p=rational(point['copy_share']);bern=sum(comb(3,k)*p**k*(1-p)**(3-k)*a[k]for k in range(4));ordinary=sum(v*p**j for j,v in enumerate(coeff))
            assert bern==ordinary==rational(point['expected_payout']);curve_points+=1
    unique_primary=list(dict.fromkeys(p['candidate_id']for p in r['primary'].values()));fresh_cells=triplet_count=primary_direct=0
    for key,model in r['fresh'].items():
        definition=model['definition'];mix=[rational(v)for v in definition['mix']]
        assert model['definition']==next(v for v in freeze['definitions']if v['key']==key)
        with np.load(out/model['file'])as a:rows=a['triplets'];types=a['family_ids'];saved=a['payout4'];castlesum=a['castle_sum12']
        regenerated,rt=fresh_triplets(mix,definition['style'],definition['size'],definition['seed']);np.testing.assert_array_equal(rows,regenerated);np.testing.assert_array_equal(types,rt)
        assert np.all(rows.sum(axis=2)==100)and np.all(rows[:,:,4][types==0]==100)and np.all(rows[:,:,4][types!=0]<100)
        calculated,c=payoff_arrays([r['candidates'][cid]['allocation']for cid in r['shortlist']],rows,candidate_batch=7,triplet_batch=10007)
        np.testing.assert_array_equal(calculated,saved);np.testing.assert_array_equal(c.sum(axis=1,dtype=np.int64),castlesum)
        for j,cid in enumerate(r['shortlist']):
            assert Fraction(int(saved[j].sum(dtype=np.int64)),4*len(rows))==rational(r['fresh_scores'][cid][key])
            assert Fraction(int(castlesum[j]),12*len(rows))==rational(r['fresh_castle_scores'][cid][key]);fresh_cells+=1
        for cid in unique_primary:
            j=r['shortlist'].index(cid);x=tuple(r['candidates'][cid]['allocation'])
            for start in range(0,len(rows),8192):
                expected=focal_samples(x,rows[start:start+8192]);np.testing.assert_array_equal(saved[j,start:start+8192],expected['payout4']);primary_direct+=len(expected['payout4'])
        triplet_count+=len(rows);print('Verified fresh Game 2 scenario: '+key,flush=True)
    for cid,values in r['fresh_scores'].items():
        v=[rational(x)for x in values.values()];s=r['fresh_summary'][cid]
        assert sum(v)/len(v)==rational(s['mean'])and min(v)==rational(s['minimum'])and max(v)==rational(s['maximum'])
    assert abs(r['sampling_bound']-fixed_sample_bound(cfg['evaluation_triplets'],len(r['shortlist'])*len(r['fresh']),cfg['sampling_alpha']))<1e-14
    for name in ['game1-selected-v1.json','game2-foundation-checkpoint-v1.json']:
        for path,digest in read(ROOT/'results'/name)['source_artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
       'candidate_field':len(actions),'anchor_triples':len(ids),'exact_anchor_candidate_cells_foundation_checked':exact_cells,'simplex_settings':len(d['mixes']),
       'frozen_primary_rules':len(r['primary']),'unique_primary_allocations':len(unique_primary),'frozen_fresh_candidates':len(r['shortlist']),
       'clone_witnesses':clone_witnesses,'cubic_curve_points_checked':curve_points,'fresh_scenarios':len(r['fresh']),'shared_fresh_triplets_regenerated':triplet_count,
       'fresh_score_cells_recomputed':fresh_cells,'primary_fresh_candidate_triplet_cells_foundation_checked':primary_direct,
       'freeze_sha256':freeze_hash,'game1_unchanged':True,'scope':'Conditional human hypotheses and first population challenge; no global optimum, equilibrium, or applicant forecast'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
