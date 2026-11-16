from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import json
import unittest

from blotto_lab.core import (
    DEFAULT_GAME, Game, Population, best_response, constructive_counter,
    exhaustive_best_response, iter_allocations, mix_tables, multinomial, score,
    score_twice, security_of, uniform_composition, validate_allocation,
)
from blotto_lab.arena import ResponseArchive, advance_history, latest_only
from blotto_lab.populations import initial_scenarios, random_composition


class RulesTests(unittest.TestCase):
    def test_allocation_validation(self):
        self.assertEqual(validate_allocation([0,0,0,0,100]), (0,0,0,0,100))
        invalid = [None, (), (20,20,20,20), (0,0,0,0,99), (-1,1,0,0,100),
                   (20.,20,20,20,20), (True,0,0,0,99), "0000100"]
        for x in invalid:
            with self.subTest(x=x), self.assertRaises(ValueError):
                validate_allocation(x)

    def test_game_validation(self):
        for args in [(-1,(1,)), (True,(1,)), (1,()), (1,(0,)), (1,(1.0,))]:
            with self.assertRaises(ValueError):
                Game(*args)

    def test_hand_checked_ties_and_zeros(self):
        x, y = (0,0,0,50,50), (0,0,100,0,0)
        self.assertEqual(score(x,y), Fraction(21,2))
        self.assertEqual(score(y,x), Fraction(9,2))
        self.assertEqual(score((0,0,0,0,100),(97,1,1,1,0)), 5)

    def test_conservation_and_self_scores(self):
        rng = Random(731)
        for _ in range(500):
            x = random_composition(rng,100,5)
            y = random_composition(rng,100,5)
            self.assertEqual(score_twice(x,y)+score_twice(y,x),30)
            self.assertEqual(score(x,x),Fraction(15,2))

    def test_castle_labels_matter(self):
        x,y = (100,0,0,0,0),(0,0,0,0,100)
        self.assertNotEqual(score(x,y),score(x[::-1],y[::-1]))

    def test_enumerator_legality_order_and_counts(self):
        for game in [Game(b,(1,2,3)) for b in range(7)]+[Game(4,(1,))]:
            actions = list(iter_allocations(game))
            self.assertEqual(len(actions),game.actions)
            self.assertEqual(actions,sorted(set(actions)))
            for x in actions:
                self.assertEqual(validate_allocation(x,game),x)
        self.assertEqual(DEFAULT_GAME.actions,4598126)


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.a,self.b=(20,20,20,20,20),(7,13,20,27,33)

    def test_duplicates_keep_their_mass(self):
        population=Population.from_people([self.a]*80+[self.b])
        self.assertEqual(population.total_mass,81)
        self.assertEqual(dict(population.entries)[self.a],80)
        x=(0,21,21,21,37)
        self.assertEqual(population.table().expected_score(x),
                         (80*score(x,self.a)+score(x,self.b))/81)

    def test_marginal_table_matches_reference(self):
        game=Game(7,(1,2,3,4,5))
        legal=list(iter_allocations(game))
        rng=Random(98)
        population=Population.from_counts([(rng.choice(legal),rng.randint(1,8)) for _ in range(31)],game)
        table=population.table()
        for x in legal:
            self.assertEqual(table.expected_score(x),population.reference_mean(x))

    def test_exclusion_removes_one_person(self):
        population=Population.from_people([self.a,self.a,self.b])
        reduced=population.excluding_one(self.a)
        self.assertEqual(reduced.total_mass,2)
        self.assertEqual(dict(reduced.entries)[self.a],1)
        with self.assertRaises(ValueError):
            reduced.excluding_one((0,0,0,0,100))
        with self.assertRaises(ValueError):
            Population.from_people([self.a]).excluding_one(self.a)

    def test_complete_round_robin_mean(self):
        people=[self.a,self.a,self.b,(0,0,0,0,100),(100,0,0,0,0)]
        population=Population.from_people(people)
        actual=sum(population.excluding_one(x).table().expected_score(x) for x in people)/len(people)
        self.assertEqual(actual,Fraction(15,2))

    def test_invalid_mass_and_empty_populations(self):
        for pairs in [[],[(self.a,0)],[(self.a,-1)],[(self.a,1.0)],[(self.a,True)]]:
            with self.assertRaises(ValueError):
                Population.from_counts(pairs)

    def test_serialization(self):
        original=Population.from_people([self.a,self.a,self.b])
        self.assertEqual(Population.from_dict(json.loads(json.dumps(original.to_dict()))),original)

    def test_exact_mixture(self):
        p=Population.from_people([self.a]*3+[self.b]).table()
        q=uniform_composition()
        mixed=mix_tables([p,q],[Fraction(2,7),Fraction(5,7)])
        for x in [self.a,self.b,(0,12,23,30,35),(0,0,0,0,100)]:
            self.assertEqual(mixed.expected_score(x),(2*p.expected_score(x)+5*q.expected_score(x))/7)
        with self.assertRaises(ValueError):
            mix_tables([p,q],[0.5,0.5])
        with self.assertRaises(ValueError):
            mix_tables([p,q],[0,0])


class OptimizerTests(unittest.TestCase):
    def test_dp_vs_exhaustive_weighted_small_games(self):
        rng=Random(20261029)
        for budget in range(9):
            game=Game(budget,(1,2,3,4,5))
            legal=list(iter_allocations(game))
            for _ in range(12):
                table=Population.from_counts([(rng.choice(legal),rng.randint(1,9)) for _ in range(7)],game).table()
                dp=best_response(table)
                scan=exhaustive_best_response(table)
                self.assertEqual((dp.expected_score,dp.allocation,dp.optimum_count),
                                 (scan.expected_score,scan.allocation,scan.optimum_count))

    def test_uniform_fixture(self):
        response=best_response(uniform_composition())
        self.assertEqual(response.allocation,(0,12,23,30,35))
        self.assertEqual(response.expected_score,Fraction(90904115,9196252))
        self.assertEqual(response.optimum_count,1)
        self.assertEqual(response.evaluations,25755)

    def test_tie_count_and_lexicographic_rule(self):
        game=Game(1,(1,1,1,1,1))
        response=best_response(uniform_composition(game))
        self.assertEqual(response.optimum_count,5)
        self.assertEqual(response.allocation,(0,0,0,0,1))
        self.assertEqual(response.expected_score,Fraction(5,2))

    def test_analytic_composition_matches_explicit_population(self):
        for game in [Game(5,(1,2,3)),Game(4,(1,2,3,4,5)),Game(3,(1,))]:
            explicit=Population.from_people(iter_allocations(game),game).table()
            analytic=uniform_composition(game)
            for x in iter_allocations(game):
                self.assertEqual(analytic.expected_score(x),explicit.expected_score(x))

    def test_multinomial_matches_independent_soldier_assignments(self):
        game=Game(4,(1,2,3))
        counts=Counter()
        for assignments in product(range(3),repeat=4):
            counts[tuple(assignments.count(i) for i in range(3))]+=1
        explicit=Population.from_counts(counts,game).table()
        analytic=multinomial(game)
        for x in iter_allocations(game):
            self.assertEqual(analytic.expected_score(x),explicit.expected_score(x))

    def test_coarse_grid_matches_explicit_grid(self):
        game=Game(6,(1,2,3))
        vectors=[x for x in iter_allocations(game) if all(k%2==0 for k in x)]
        explicit=Population.from_people(vectors,game).table()
        analytic=uniform_composition(game,unit=2)
        for x in iter_allocations(game):
            self.assertEqual(analytic.expected_score(x),explicit.expected_score(x))

    def test_constructive_counters_and_security(self):
        rng=Random(999)
        for x in [(0,0,100,0,0),(100,0,0,0,0),(0,0,0,0,100)]+[random_composition(rng,100,5) for _ in range(200)]:
            counter=constructive_counter(x)
            self.assertGreaterEqual(score(counter,x),10)
            info=security_of(Population.from_people([x]))
            self.assertLessEqual(info['security'],5)
        self.assertEqual(security_of(Population.from_people([(0,0,0,0,100)]))['security'],5)


class ArenaTests(unittest.TestCase):
    def test_resume_is_identical_to_uninterrupted_history(self):
        game=Game(5,(1,2,3))
        continuous=ResponseArchive.start((1,2,2),game)
        advance_history(continuous,35,record_every=7)
        split=ResponseArchive.start((1,2,2),game)
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'state.json'
            advance_history(split,13,checkpoint=checkpoint,checkpoint_every=5)
            resumed=ResponseArchive.load(checkpoint)
            trace=advance_history(resumed,35,record_every=7)
        self.assertEqual(resumed.counts,continuous.counts)
        self.assertEqual(resumed.last_added,continuous.last_added)
        self.assertEqual(trace[-1]['security']['decimal'],float(security_of(continuous.population())['security']))

    def test_bad_checkpoint_rejected(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.json'
            ResponseArchive.start().save(path)
            data=json.loads(path.read_text())
            data['steps']=100
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ResponseArchive.load(path)

    def test_synthetic_scenarios_reproducible(self):
        a=initial_scenarios(123,30)
        b=initial_scenarios(123,30)
        self.assertEqual([s.table.numerators for s in a],[s.table.numerators for s in b])
        self.assertEqual([s.population for s in a],[s.population for s in b])

    def test_checkpoint_source_change_rejected(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.json'
            ResponseArchive.start().save(path)
            data=json.loads(path.read_text())
            data['implementation_sha256']='changed'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ResponseArchive.load(path)

    def test_latest_only_replies_really_improve(self):
        result=latest_only(max_steps=20)
        for step in result['trace']:
            self.assertGreater(step['counter_score']['decimal'],7.5)


if __name__ == '__main__':
    unittest.main()
