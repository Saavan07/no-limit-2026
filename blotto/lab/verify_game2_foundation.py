"""Independent rational group scoring, people-level enumeration, and source audit."""
from fractions import Fraction
from itertools import combinations,product,permutations,islice
from pathlib import Path
import argparse,json,time
import numpy as np
from blotto_lab.core import Game,Population,iter_allocations
from game2_lab.core import (reference_group,group_scores12_4,shared_triplets,evaluate_shared,exact_evaluation)
from run_multiwave_study import ROOT
from run_population_study import save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def audit_group(rows,values=(1,2,3,4,5)):
    totals=[Fraction(0)for _ in range(4)]
    for castle,value in enumerate(values):
        buckets={}
        for player,row in enumerate(rows):buckets.setdefault(int(row[castle]),[]).append(player)
        winners=buckets[max(buckets)]
        for player in winners:totals[player]+=Fraction(value,len(winners))
    payout=[None]*4;position=0;prizes=(9,4,2,0)
    for total in sorted(set(totals),reverse=True):
        tied=[p for p in range(4)if totals[p]==total]
        reward=Fraction(sum(prizes[position:position+len(tied)]),len(tied))
        for player in tied:payout[player]=reward
        position+=len(tied)
    return tuple(totals),tuple(payout)


def direct_evaluation(x,pop,scheme):
    people=[a for a,m in pop.entries for _ in range(m)]
    ids=product(range(len(people)),repeat=3)if scheme=='iid'else combinations(range(len(people)),3)
    total=Fraction(0);castle=Fraction(0);count=0
    for triple in ids:
        c,p=audit_group([x]+[people[i]for i in triple]);total+=p[0];castle+=c[0];count+=1
    return total/count,castle/count,count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-foundation')
    out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');m=read(out/'manifest.json')
    for name,digest in m['artifacts'].items():assert sha(out/name)==digest,name
    for name,digest in m['signature']['source_hashes'].items():assert sha(ROOT/name)==sha(out/'source-snapshot'/name)==digest,name
    for name,digest in m['signature']['input_hashes'].items():assert sha(ROOT/name)==digest,name
    game=Game(2,(1,2,3,4,5));actions=list(iter_allocations(game));iterator=product(actions,repeat=4);profiles=0
    while True:
        batch=list(islice(iterator,1024))
        if not batch:break
        c,p=group_scores12_4(np.asarray(batch),game)
        for j,group in enumerate(batch):
            expected_c,expected_p=audit_group(group,game.values);ref=reference_group(group,game)
            assert ref.castle_points==expected_c and ref.payouts==expected_p
            assert tuple(Fraction(int(v),12)for v in c[j])==expected_c and tuple(Fraction(int(v),4)for v in p[j])==expected_p
        profiles+=len(batch)
    assert profiles==50625
    rng=np.random.default_rng(8272026);groups=rng.multinomial(100,rng.dirichlet(np.ones(5),size=(100*4))).reshape(100,4,5)
    c,p=group_scores12_4(groups);permutation_checks=0
    for j,group in enumerate(groups):
        expected_c,expected_p=audit_group(group);assert tuple(Fraction(int(v),12)for v in c[j])==expected_c and tuple(Fraction(int(v),4)for v in p[j])==expected_p
        if j<8:
            for perm in permutations(range(4)):
                rotated=group[list(perm)];cc,pp=group_scores12_4(rotated[None,:,:]);np.testing.assert_array_equal(cc[0],c[j,list(perm)]);np.testing.assert_array_equal(pp[0],p[j,list(perm)]);permutation_checks+=1
    exact_cells=sample_cells=sample_triplets=direct_sample_checks=0;maximum_error=Fraction(0)
    for name,definition in r['populations'].items():
        pop=Population.from_dict(definition)
        for scheme in ['iid','finite']:
            sample=r['sampled'][name][scheme];filename=out/sample['sample_file']
            with np.load(filename)as a:
                rows=a['triplets'];types=a['type_ids'];people=a['person_ids']if scheme=='finite'else None
            regenerated=shared_triplets(pop,sample['size'],sample['seed'],scheme);np.testing.assert_array_equal(rows,regenerated['triplets']);np.testing.assert_array_equal(types,regenerated['type_ids'])
            if scheme=='finite':
                np.testing.assert_array_equal(people,regenerated['person_ids']);assert np.all((people>=0)&(people<pop.total_mass))
                assert all(len(set(v))==3 for v in people)
                expanded=[a for a,n in pop.entries for _ in range(n)];np.testing.assert_array_equal(rows,np.array(expanded)[people])
            samples=evaluate_shared([tuple(x)for x in r['candidates'].values()],rows,batch_size=703)
            for (label,x),estimate in zip(r['candidates'].items(),samples):
                payout,castle,count=direct_evaluation(tuple(x),pop,scheme);saved=r['exact'][name][scheme][label]
                assert payout==rational(saved['expected_payout'])and castle==rational(saved['expected_castle_points'])and count==saved['probability_denominator'];exact_cells+=1
                recorded=sample['evaluations'][label];assert estimate.expected_payout==rational(recorded['expected_payout'])and estimate.expected_castle_points==rational(recorded['expected_castle_points']);sample_cells+=1
                maximum_error=max(maximum_error,abs(estimate.expected_payout-payout))
                # A separate group scorer checks a fixed subset of sampled outcomes.
                expected=sum(audit_group([x]+list(triple))[1][0]for triple in rows[:20])/20
                first20=evaluate_shared([tuple(x)],rows[:20])[0];assert expected==first20.expected_payout;direct_sample_checks+=20
            sample_triplets+=len(rows)
        iid=sum(n*exact_evaluation(x,pop,'iid').expected_payout for x,n in pop.entries)/pop.total_mass
        finite=sum(n*exact_evaluation(x,pop.excluding_one(x),'finite').expected_payout for x,n in pop.entries)/pop.total_mass
        assert iid==finite==Fraction(15,4)
    for example in r['five_points_examples'].values():
        c,p=audit_group(example['group']);assert list(c)==[rational(v)for v in example['outcome']['castle_points']]
        assert list(p)==[rational(v)for v in example['outcome']['payouts']]
    joint=r['joint_counterexample'];a=Population.from_dict(joint['population_p']);b=Population.from_dict(joint['population_q']);x=tuple(joint['candidate'])
    assert a.histograms()==b.histograms()
    for ar,br in zip(a.table().numerators,b.table().numerators):assert all(v*b.table().denominator==w*a.table().denominator for v,w in zip(ar,br))
    pa,ca,_=direct_evaluation(x,a,'iid');pb,cb,_=direct_evaluation(x,b,'iid');assert ca==cb and pa!=pb
    assert pa==rational(joint['game2_p']['expected_payout'])and pb==rational(joint['game2_q']['expected_payout'])
    reversal=r['objective_reversal'];pop=Population.from_dict(reversal['population']);pa,ca,_=direct_evaluation(tuple(reversal['higher_castle_candidate']),pop,'iid');pb,cb,_=direct_evaluation(tuple(reversal['higher_payout_candidate']),pop,'iid')
    assert ca>cb and pa<pb
    assert pa==rational(reversal['higher_castle_evaluation']['expected_payout'])and pb==rational(reversal['higher_payout_evaluation']['expected_payout'])
    for filename in ['game1-selected-v1.json','game1-final-review-checkpoint-v1.json','game1-merged-checkpoint-v2.json','game1-machine-checkpoint-v1.json']:
        for name,digest in read(ROOT/'results'/filename)['source_artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
       'exhaustive_reduced_game_profiles':profiles,'default_game_groups_checked':len(groups),'player_permutations_checked':permutation_checks,
       'exact_iid_and_finite_control_cells':exact_cells,'sample_evaluation_cells':sample_cells,'shared_triplets_regenerated':sample_triplets,
       'independent_sample_group_crosschecks':direct_sample_checks,'maximum_control_sample_error':maximum_error,
       'identical_marginals_different_rank_payout_verified':True,'castle_point_objective_reversal_verified':True,
       'symmetric_iid_and_finite_calibration':'15/4','game1_recommendation_unchanged':True,
       'scope':'Game 2 exact scoring foundation and declared small controls, no strategy optimum or applicant forecast'}
    save(out/'acceptance.json',acceptance);print(json.dumps(__import__('run_population_study').encode(acceptance),indent=2))


if __name__=='__main__':main()
