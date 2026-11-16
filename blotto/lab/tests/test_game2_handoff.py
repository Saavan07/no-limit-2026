from fractions import Fraction
import unittest
import numpy as np
from blotto_lab.core import Game,Population,iter_allocations
from game2_lab.core import exact_evaluation,reference_group
from game2_lab.handoff import critical_actions,critical_count,exact_critical_response,target_population,cdf_diagnostic


class HandoffTests(unittest.TestCase):
    def test_critical_response_matches_all_legal_small_actions(self):
        rng=np.random.default_rng(301)
        for game in [Game(4,(1,2,3)),Game(2,(1,2,3,4,5))]:
            actions=list(iter_allocations(game))
            for _ in range(8):
                pop=Population.from_people([actions[int(i)]for i in rng.integers(len(actions),size=3)],game)
                actual=exact_critical_response(pop);best=max(exact_evaluation(x,pop).expected_payout for x in actions)
                self.assertEqual(actual['payout'],best)
                for slack in range(game.castles):
                    reduced=list(critical_actions([x for x,n in pop.entries],game,slack));self.assertEqual(len(reduced),critical_count([x for x,n in pop.entries],slack,game));self.assertEqual(len(reduced),len(set(reduced)))

    def test_identical_incumbent_scores_do_not_identify_new_entry_payoff(self):
        y=(0,0,0,50,50);transformed=tuple(4*v//5+4 for v in y);x=(30,30,0,20,20)
        self.assertEqual(reference_group([y]*4).payouts,reference_group([transformed]*4).payouts)
        self.assertEqual(reference_group([x,y,y,y]).payouts[0],Fraction(15,4));self.assertEqual(reference_group([x,transformed,transformed,transformed]).payouts[0],0)

    def test_identical_marginals_different_joint_rank_scores(self):
        q1=Population.from_people([(40,10,40,10,0),(10,40,10,40,0)]);q2=Population.from_people([(10,40,40,10,0),(40,10,10,40,0)]);x=(0,0,100,0,0)
        self.assertEqual(exact_evaluation(x,q1).expected_payout,Fraction(45,8));self.assertEqual(exact_evaluation(x,q2).expected_payout,Fraction(57,8))
        self.assertEqual(exact_evaluation(x,q1).expected_castle_points,Fraction(17,4));self.assertEqual(exact_evaluation(x,q2).expected_castle_points,Fraction(17,4))

    def test_target_families_cover_full_active_integer_support(self):
        from math import comb
        for active in [(1,3),(0,2,4)]:
            for scout in [0,3,10]:
                expected=comb(100-(5-len(active))*scout-8*len(active)+len(active)-1,len(active)-1)
                for mode in ['uniform','equal','value']:
                    pop=target_population(active,scout,mode);self.assertEqual(len(pop.entries),expected)
                    for x,n in pop.entries:
                        self.assertEqual(sum(x),100);self.assertGreater(n,0)
                        self.assertTrue(all(x[i]>=8 if i in active else x[i]==scout for i in range(5)))

    def test_cdf_metric_includes_left_limits(self):
        a=cdf_diagnostic([1,4],[1,1],[1,4]);self.assertEqual(a['maximum_cdf_gap'],0)
        b=cdf_diagnostic([2,3],[1,1],[1,4]);self.assertEqual(b['maximum_cdf_gap'],.5)


if __name__=='__main__':unittest.main()
