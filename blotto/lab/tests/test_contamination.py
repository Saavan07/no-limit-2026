from fractions import Fraction
from random import Random
import unittest
import numpy as np
from blotto_lab.core import Game, Population, best_response, iter_allocations, score, uniform_composition
from blotto_lab.contamination import (FrontierObjective, continuous_envelope, exact_counter,
    objective_value, pattern_security_twice, search_frontiers, security_twice, select_epsilon)
from blotto_lab.populations import random_composition


class ContaminationTests(unittest.TestCase):
    def test_structural_security_and_pattern_enumerator_against_dp_complete_tiny_games(self):
        for game in [Game(0,(1,2,3)),Game(6,(1,2,3)),Game(5,(1,2,3,4,5)),Game(3,(1,)),Game(5,(3,1,4))]:
            actions = list(iter_allocations(game))
            primary = security_twice(actions,game)
            independent = pattern_security_twice(actions,game)
            np.testing.assert_array_equal(primary,independent)
            for x,w in zip(actions,primary):
                br = best_response(Population.from_people([x],game).table())
                self.assertEqual(Fraction(int(w),2),game.total_value-br.expected_score)
                y,guarantee = exact_counter(x,game)
                self.assertEqual(score(x,y,game),guarantee)
                self.assertEqual(guarantee,Fraction(int(w),2))

    def test_real_game_counter_witnesses(self):
        rng = Random(808)
        actions = [random_composition(rng,100,5) for _ in range(120)]+[(0,0,0,0,100),(4,19,27,48,2)]
        for x,w in zip(actions,security_twice(actions)):
            y,value = exact_counter(x)
            self.assertEqual(score(x,y),value)
            self.assertEqual(15-best_response(Population.from_people([x]).table()).expected_score,value)
            self.assertEqual(value,Fraction(int(w),2))
        with self.assertRaises(ValueError):
            security_twice([[20.0]*5])

    def test_conditional_global_frontiers_and_contamination_against_fraction_bruteforce(self):
        game = Game(7,(1,2,3))
        actions = list(iter_allocations(game))
        tables = [uniform_composition(game),Population.from_people([(0,2,5),(1,3,3),(2,0,5)],game).table()]
        objectives = [FrontierObjective('score',(0,1)),FrontierObjective('regret',(0,1),'regret')]
        actual = search_frontiers(tables,objectives,batch_size=5)
        optima = [best_response(t).expected_score for t in tables]
        securities = {x:game.total_value-best_response(Population.from_people([x],game).table()).expected_score for x in actions}
        for obj in objectives:
            for c,p in actual['frontiers'][obj.name].items():
                values = [(objective_value(x,tables,obj,optima),x) for x in actions if 2*securities[x]==c]
                maximum = max(v for v,x in values)
                winners = [x for v,x in values if v==maximum]
                self.assertEqual(p['nominal_value'],maximum)
                self.assertEqual(p['allocation'],min(winners))
                self.assertEqual(p['optimum_count'],len(winners))
        frontier = actual['frontiers']['score']
        envelope = continuous_envelope(frontier,actual['class_counts'],actual['class_first'])
        epsilons = {Fraction(i,100) for i in range(101)}|{p['epsilon'] for p in envelope['boundary_ties']}
        for epsilon in epsilons:
            values = [((1-epsilon)*min(t.expected_score(x) for t in tables)+epsilon*securities[x],x) for x in actions]
            maximum = max(v for v,x in values)
            winners = [x for v,x in values if v==maximum]
            selected = select_epsilon(frontier,epsilon,actual['class_counts'],actual['class_first'])
            self.assertEqual(selected['value'],maximum)
            self.assertEqual(selected['allocation'],min(winners))
            self.assertEqual(selected['optimum_count'],len(winners))

    def test_all_security_vectors_tie_at_epsilon_one(self):
        game = Game(3,(1,2,3))
        result = search_frontiers([uniform_composition(game)],[FrontierObjective('one',(0,))],batch_size=3)
        selected = select_epsilon(result['frontiers']['one'],1,result['class_counts'],result['class_first'])
        maximum = max(result['class_counts'])
        self.assertEqual(selected['optimum_count'],result['class_counts'][maximum])
        self.assertEqual(selected['allocation'],result['class_first'][maximum])


if __name__=='__main__':
    unittest.main()
