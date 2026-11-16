from fractions import Fraction
from pathlib import Path
from random import Random
import unittest

import numpy as np

from blotto_lab.core import Game, Population, best_response, iter_allocations, uniform_composition
from blotto_lab.historical import (fit_features, load_round, round_rows_to_grid,
    rows_to_table, sample_synthetic, sample_transferred, transfer_parameters, validate_rows)
from blotto_lab.robust import Objective, exact_objective, float_tables, search
from run_equilibrium_benchmark import centered_matrix, integer_masses, solve_restricted


class RobustTests(unittest.TestCase):
    def test_restricted_matrix_and_equilibrium_on_complete_tiny_game(self):
        from blotto_lab.core import score_twice, security_of
        game = Game(4,(1,2,3))
        actions = list(iter_allocations(game))
        matrix = centered_matrix(actions,game)
        for i,x in enumerate(actions):
            for j,y in enumerate(actions):
                self.assertEqual(matrix[i,j],score_twice(x,y,game)-6)
        population,diagnostics = solve_restricted(actions,game)
        self.assertLess(security_of(population)['exploitability'],Fraction(1,10**10))
        self.assertEqual(sum(integer_masses([.1,.2,.7])),10**15)
        self.assertEqual(population.total_mass,10**15)

    def test_global_selection_and_ties_against_fraction_enumeration(self):
        rng = Random(7701)
        for budget in (0,1,4,7):
            game = Game(budget,(1,2,3))
            actions = list(iter_allocations(game))
            tables = [uniform_composition(game)] + [Population.from_counts(
                [(rng.choice(actions),rng.randint(1,19)) for _ in range(9)],game).table()
                for _ in range(3)]
            objectives = [Objective('score',(0,1,2,3)),
                          Objective('regret',(0,1,2,3),'minimax_regret')]
            optima = [best_response(t).expected_score for t in tables]
            actual = search(tables,objectives,batch_size=5)
            for obj in objectives:
                values = [(exact_objective(x,tables,obj.model_indices,obj.mode,optima),x) for x in actions]
                maximum = max(v for v,x in values)
                winners = [x for v,x in values if v==maximum]
                self.assertEqual(actual[obj.name]['allocation'],min(winners))
                self.assertEqual(actual[obj.name]['maximized_value'],maximum)
                self.assertEqual(actual[obj.name]['optimum_count'],len(winners))
                self.assertEqual(actual[obj.name]['evaluated'],game.actions)

    def test_error_enclosure_for_every_score_in_small_game(self):
        game = Game(8,(1,3,5))
        tables = [uniform_composition(game), Population.from_counts(
            [((1,3,4),17),((4,3,1),29),((0,8,0),101)],game).table()]
        floating, errors = float_tables(tables)
        for x in iter_allocations(game):
            for i,t in enumerate(tables):
                value = np.float64(0)
                for j,k in enumerate(x):
                    value += floating[i,j,k]
                self.assertLessEqual(abs(Fraction.from_float(float(value))-t.expected_score(x)),errors[i])

    def test_single_scenario_agrees_with_exact_dp(self):
        game = Game(9,(1,1,1,1,1))
        table = uniform_composition(game)
        expected = best_response(table)
        result = search([table],[Objective('one',(0,))],batch_size=7)['one']
        self.assertEqual(result['allocation'],expected.allocation)
        self.assertEqual(result['optimum_count'],expected.optimum_count)
        self.assertEqual(result['maximized_value'],expected.expected_score)


class GeneratorTests(unittest.TestCase):
    def test_synthetic_legality_determinism_and_structure(self):
        for family in ('equal_jitter','value_jitter','sparse','middle_focus','top_focus','one_up_grid'):
            rows = sample_synthetic(family,2000,737)
            validate_rows(rows)
            np.testing.assert_array_equal(rows,sample_synthetic(family,2000,737))
            if family=='sparse':
                self.assertTrue(np.all((rows>0).sum(axis=1)>=2))
                self.assertTrue(np.all((rows>0).sum(axis=1)<=4))
            if family=='middle_focus':
                self.assertTrue(np.all(rows[:,[0,4]]<=3))
            if family=='top_focus':
                self.assertTrue(np.all(rows[:,:3]<=3))

    def test_largest_remainder_rounding(self):
        rows = sample_synthetic('sparse',1000,1)
        rounded = round_rows_to_grid(rows,np.random.default_rng(9))
        self.assertTrue(np.all(rounded%5==0))
        self.assertTrue(np.all(np.abs(rounded-rows)<5))
        validate_rows(rounded)

    def test_table_matches_direct_scoring(self):
        rows = sample_synthetic('sparse',20,1)
        population = Population.from_people([tuple(map(int,r)) for r in rows])
        table = rows_to_table(rows,'check')
        for x in [(0,0,0,0,100),(20,20,20,20,20),(0,12,23,30,35)]:
            self.assertEqual(table.expected_score(x),population.reference_mean(x))

    def test_feature_moment_recovery_and_transfers(self):
        rng = np.random.default_rng(901)
        probabilities = np.arange(1,11)/55
        rows = rng.multinomial(100,rng.dirichlet(6*probabilities,size=100000))
        features = fit_features(rows)
        self.assertAlmostEqual(features['dirichlet_total_alpha'],6,delta=.10)
        self.assertLess(np.max(np.abs(np.asarray(features['mean_shares'])-probabilities)),.002)
        for profile in ('power','rank_curve'):
            for concentration in ('fixed_total','fixed_per_field'):
                parameters = transfer_parameters(features,profile,concentration)
                self.assertAlmostEqual(sum(parameters['probabilities']),1)
                draws = sample_transferred(parameters,500,2)
                validate_rows(draws)
                np.testing.assert_array_equal(draws,sample_transferred(parameters,500,2))

    def test_pinned_historical_data(self):
        folder = Path(__file__).resolve().parents[1]/'data/riddler'
        for number,records in [(1,1349),(2,902),(3,1321),(4,938),(5,895)]:
            rows,metadata = load_round(folder,number)
            self.assertEqual(len(rows),records)
            self.assertEqual(metadata['castles'],13 if number==5 else 10)


if __name__=='__main__':
    unittest.main()
