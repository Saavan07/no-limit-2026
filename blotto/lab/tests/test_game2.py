from fractions import Fraction
from itertools import combinations,permutations,product
from pathlib import Path
import unittest
import numpy as np
from blotto_lab.core import Game,Population,iter_allocations
from game2_lab.core import (reference_group,rank_payouts4,group_scores12_4,focal_samples,
    exact_evaluation,shared_triplets,evaluate_shared,fixed_sample_bound)


class Game2ScoringTests(unittest.TestCase):
    def test_every_rank_tie_pattern(self):
        cases=[([9,3,2,1],[36,16,8,0]),([5,5,3,2],[26,26,8,0]),([7,4,4,0],[36,12,12,0]),
               ([8,5,1,1],[36,16,4,4]),([4,4,4,3],[20,20,20,0]),([6,3,3,3],[36,8,8,8]),([4,4,4,4],[15,15,15,15])]
        for scores,expected in cases:np.testing.assert_array_equal(rank_payouts4(np.array([scores]))[0],expected)

    def test_castle_splits_and_identical_players(self):
        x=(20,20,20,20,20);out=reference_group([x]*4)
        self.assertEqual(out.castle_points,(Fraction(15,4),)*4);self.assertEqual(out.payouts,(Fraction(15,4),)*4)
        groups=[[(100,0,0,0,0)]*k+[(0,100,0,0,0)]*(4-k)for k in [1,2,3]]
        for g in groups:
            expected=reference_group(g);c,p=group_scores12_4(np.array([g]))
            self.assertEqual(tuple(Fraction(int(v),12)for v in c[0]),expected.castle_points)
            self.assertEqual(tuple(Fraction(int(v),4)for v in p[0]),expected.payouts)

    def test_exhaustive_tiny_profiles_against_fraction_reference(self):
        game=Game(2,(1,2,3));actions=list(iter_allocations(game));groups=np.array(list(product(actions,repeat=4)))
        c,p=group_scores12_4(groups,game)
        for i,g in enumerate(groups):
            ref=reference_group([tuple(map(int,a))for a in g],game)
            self.assertEqual(tuple(Fraction(int(v),12)for v in c[i]),ref.castle_points)
            self.assertEqual(tuple(Fraction(int(v),4)for v in p[i]),ref.payouts)

    def test_player_permutations_and_invalid_inputs(self):
        group=[(0,0,0,0,100),(0,0,0,100,0),(0,0,100,0,0),(50,50,0,0,0)];base=reference_group(group)
        for perm in permutations(range(4)):
            result=reference_group([group[i]for i in perm]);self.assertEqual(result.payouts,tuple(base.payouts[i]for i in perm))
        with self.assertRaises(ValueError):reference_group(group[:3])
        with self.assertRaises(ValueError):group_scores12_4(np.array([group],dtype=float))
        with self.assertRaises(ValueError):group_scores12_4(np.array([[[20,20,20,20,19]]*4]))


class Game2PopulationTests(unittest.TestCase):
    def people(self):return [(20,20,20,20,20)]*3+[(0,0,50,0,50)]*2+[(0,0,0,50,50)]

    def test_finite_compression_matches_distinct_people_enumeration(self):
        people=self.people();pop=Population.from_people(people);x=(3,8,4,44,41)
        exact=exact_evaluation(x,pop,'finite');triples=list(combinations(range(len(people)),3))
        pay=sum(reference_group([x]+[people[i]for i in ids]).payouts[0]for ids in triples)/len(triples)
        self.assertEqual(exact.expected_payout,pay);self.assertEqual(exact.probability_denominator,20)
        self.assertEqual(sum(exact.finish_credits),1)

    def test_iid_compression_matches_ordered_support_enumeration(self):
        people=self.people();pop=Population.from_people(people);x=(3,8,4,44,41)
        exact=exact_evaluation(x,pop,'iid')
        pay=sum(reference_group([x]+[people[i]for i in ids]).payouts[0]for ids in product(range(len(people)),repeat=3))/len(people)**3
        self.assertEqual(exact.expected_payout,pay)

    def test_symmetric_population_calibrates_to_3_75(self):
        pop=Population.from_people(self.people());n=pop.total_mass
        iid=sum(m*exact_evaluation(a,pop,'iid').expected_payout for a,m in pop.entries)/n
        finite=sum(m*exact_evaluation(a,pop.excluding_one(a),'finite').expected_payout for a,m in pop.entries)/n
        self.assertEqual(iid,Fraction(15,4));self.assertEqual(finite,Fraction(15,4))

    def test_shared_sampling_reproducibility_and_no_duplicate_people(self):
        pop=Population.from_people(self.people())
        for scheme in ['iid','finite']:
            a=shared_triplets(pop,400,77,scheme);b=shared_triplets(pop,400,77,scheme)
            np.testing.assert_array_equal(a['triplets'],b['triplets'])
            if scheme=='finite':self.assertTrue(all(len(set(row))==3 for row in a['person_ids']))
        x=(3,8,4,44,41);a=shared_triplets(pop,50,8)
        first=evaluate_shared([x],a['triplets'],batch_size=7)[0];second=evaluate_shared([x],a['triplets'],batch_size=50)[0]
        self.assertEqual(first,second)
        with self.assertRaises(ValueError):shared_triplets(Population.from_people(self.people()[:2]),10,2,'finite')

    def test_five_points_can_win_or_place_second(self):
        focal=(0,0,0,0,100)
        first=reference_group([focal,(0,0,0,100,0),(0,0,100,0,0),(50,50,0,0,0)])
        second=reference_group([focal,(0,0,50,50,0),(0,100,0,0,0),(100,0,0,0,0)])
        self.assertEqual(first.castle_points[0],5);self.assertEqual(second.castle_points[0],5)
        self.assertEqual(first.payouts[0],9);self.assertEqual(second.payouts[0],4)

    def test_fixed_sample_bounds_and_validation(self):
        self.assertGreater(fixed_sample_bound(1000,paired=True),fixed_sample_bound(1000))
        self.assertLess(fixed_sample_bound(10000),fixed_sample_bound(1000))
        with self.assertRaises(ValueError):fixed_sample_bound(0)


if __name__=='__main__':unittest.main()
