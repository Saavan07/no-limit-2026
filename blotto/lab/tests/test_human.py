from fractions import Fraction
from math import comb
import unittest
import numpy as np
from blotto_lab.core import DEFAULT_GAME, Population, score, uniform_composition, multinomial
from blotto_lab.contamination import security_twice
from blotto_lab.human import (coalition_anchor,coalition_population,joint_metrics,
    learning_path,marginal_counts,minimal_winning_coalitions,ranged_allocations,score_matrices)


class HumanChallengeTests(unittest.TestCase):
    def test_literal_ranges_and_minimal_winning_sets(self):
        rows=ranged_allocations([(2,3),(5,7),(22,25),(36,42),(25,30)])
        self.assertEqual(len(rows),114)
        self.assertEqual({score(x,(4,19,27,48,2)) for x in rows},{Fraction(5)})
        self.assertTrue(np.all(security_twice(rows)==4))
        self.assertEqual(score((0,10,30,30,30),(4,19,27,48,2)),8)
        self.assertEqual(set(minimal_winning_coalitions()),{(2,4),(3,4),(0,1,4),(0,2,3),(1,2,3)})

    def test_exact_coalition_laws_and_conditioning(self):
        for target in [(2,4),(0,2,3),(0,1,4)]:
            for scout in [0,1,3]:
                free=100-scout*(5-len(target))-len(target)
                uniform=coalition_population(target,scout,'uniform')
                weighted=coalition_population(target,scout,'value_multinomial')
                self.assertEqual(uniform.total_mass,comb(free+len(target)-1,len(target)-1))
                self.assertEqual(weighted.total_mass,sum(DEFAULT_GAME.values[i] for i in target)**free)
                self.assertEqual({x for x,m in uniform.entries},{x for x,m in weighted.entries})
                for x,m in uniform.entries:
                    self.assertEqual(sum(x),100)
                    self.assertTrue(all(x[i]>0 for i in target))
                    self.assertTrue(all(x[i]==scout for i in range(5) if i not in target))
                if 4 in target:
                    heavy=coalition_population(target,scout,'value_multinomial',50)
                    original=dict(weighted.entries)
                    self.assertTrue(all(x[4]>=50 and original[x]==m for x,m in heavy.entries))
                    self.assertLess(heavy.total_mass,weighted.total_mass)

    def test_marginals_recovered_exactly(self):
        population=Population.from_people([(1,1,20,8,70),(3,3,2,41,51),(0,10,30,30,30)])
        for table in [population.table(),uniform_composition(),multinomial()]:
            for i in range(5):
                counts,den=marginal_counts(table,i)
                self.assertEqual(sum(counts),den)
                self.assertGreaterEqual(min(counts),0)
                before=0
                for k,c in enumerate(counts):
                    self.assertEqual(Fraction(table.numerators[i][k],table.denominator),Fraction((i+1)*(2*before+c),2*den))
                    before+=c

    def test_win_and_point_metrics_match_reference(self):
        population=coalition_population((2,4),1,'value_multinomial')
        actions=[(1,1,20,8,70),(3,3,2,41,51),(0,10,30,30,30)]
        metrics=joint_metrics(actions,population,actions)
        for x in actions:
            self.assertEqual(metrics[x]['joint_mean_points'],population.table().expected_score(x))
            numerator=sum(m*(2*(score(x,y)>Fraction(15,2))+(score(x,y)==Fraction(15,2))) for y,m in population.entries)
            self.assertEqual(metrics[x]['win_tie_adjusted'],Fraction(numerator,2*population.total_mass))
        points,wins=score_matrices(actions)
        for i,x in enumerate(actions):
            for j,y in enumerate(actions):
                self.assertEqual(points[i,j],2*score(x,y))
                self.assertEqual(wins[i,j],2*(score(x,y)>Fraction(15,2))+(score(x,y)==Fraction(15,2)))

    def test_learning_rules_and_constant_payoff_rest_point(self):
        x,y=(20,20,20,20,20),(19,19,22,20,20)
        matrix,_=score_matrices([x,y])
        self.assertTrue(np.all(matrix==15))
        p=np.array([.9,.1])
        replicator=learning_path(matrix/2,p,'replicator',0,100)
        replacement=learning_path(matrix/2,p,'payoff_replacement',0,100)
        np.testing.assert_allclose(replicator['final'],p)
        np.testing.assert_allclose(replacement['final'],[.5,.5])
        mutation=learning_path(matrix/2,p,'replicator',.01,100)
        self.assertLess(mutation['final'][0],.9)
        self.assertAlmostEqual(sum(mutation['final']),1)

    def test_coalition_anchors_and_reproducible_learning(self):
        for coalition in minimal_winning_coalitions():
            for scout in [0,1,3]:
                x=coalition_anchor(coalition,scout)
                self.assertEqual(sum(x),100)
                self.assertTrue(all(x[i]==scout for i in range(5) if i not in coalition))
        points,_=score_matrices([(1,1,20,8,70),(3,3,2,41,51),(0,10,30,30,30)])
        a=learning_path(points/2,[.2,.3,.5],'replicator',.01,200,25)
        b=learning_path(points/2,[.2,.3,.5],'replicator',.01,200,25)
        np.testing.assert_array_equal(a['final'],b['final'])
        self.assertEqual(a['trace'],b['trace'])


if __name__=='__main__':unittest.main()
