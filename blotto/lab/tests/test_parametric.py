from fractions import Fraction
from random import Random
import unittest
from blotto_lab.core import Game, Population, best_response, iter_allocations, mix_tables, uniform_composition
from blotto_lab.parametric import strategic_share_map


class ParametricTests(unittest.TestCase):
    def test_complete_map_matches_exhaustive_fraction_values_and_ties(self):
        rng = Random(991)
        for budget in [2,5,7]:
            game = Game(budget,(1,2,3))
            actions = list(iter_allocations(game))
            a = Population.from_counts([(rng.choice(actions),rng.randint(1,10)) for _ in range(9)],game).table()
            b = Population.from_counts([(rng.choice(actions),rng.randint(1,10)) for _ in range(9)],game).table()
            result = strategic_share_map(a,b)
            points = {Fraction(i,100) for i in range(101)}|{p['share'] for p in result['boundaries']}
            for s in points:
                reference = best_response(mix_tables([a,b],[1-s,s]))
                boundary = next((p for p in result['boundaries'] if p['share']==s),None)
                if boundary:
                    self.assertEqual(boundary['allocation'],reference.allocation)
                    self.assertEqual(boundary['optimum_count'],reference.optimum_count)
                else:
                    segment = next(p for p in result['segments'] if p['lower_share']<s<p['upper_share'])
                    self.assertEqual(segment['allocation'],reference.allocation)
                    self.assertEqual(segment['open_interval_optimum_count'],reference.optimum_count)
                brute = max((1-s)*a.expected_score(x)+s*b.expected_score(x) for x in actions)
                self.assertEqual(brute,reference.expected_score)

    def test_identical_references_and_endpoint_ties(self):
        for game in [Game(0,(1,2,3)),Game(1,(1,1,1)),Game(5,(1,2,3))]:
            table = uniform_composition(game)
            result = strategic_share_map(table,table)
            self.assertEqual(len(result['segments']),1)
            self.assertEqual(result['segments'][0]['lower_share'],0)
            self.assertEqual(result['segments'][0]['upper_share'],1)
            self.assertEqual(result['segments'][0]['open_interval_optimum_count'],best_response(table).optimum_count)


if __name__=='__main__': unittest.main()
