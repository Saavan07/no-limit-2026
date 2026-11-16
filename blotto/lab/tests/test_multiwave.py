from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from blotto_lab.core import Game, Population, best_response, iter_allocations, uniform_composition
from blotto_lab.historical import validate_rows
from blotto_lab.multiwave import hypothesis_rows, mutate, run_path, uniform_best_response


class ReasoningTests(unittest.TestCase):
    def test_all_hypothesis_constructions_legal_and_reproducible(self):
        families = ['subset_value','cheap_scouts','fortress_five','skip_one','round_tens',
            'anchor_snipers','heterogeneous_power','monotone_ladder','inverse_value',
            'pair_commitment','grid_three','rotating_templates','budget_blocks','even_thresholds']
        for family in families:
            rows = hypothesis_rows(family,1500,770)
            validate_rows(rows)
            np.testing.assert_array_equal(rows,hypothesis_rows(family,1500,770))
            if family=='fortress_five':
                self.assertTrue(np.all(rows[:,4]>=68))
            if family=='skip_one':
                self.assertTrue(np.all((rows==0).sum(axis=1)==1))
            if family=='round_tens':
                self.assertTrue(np.all(rows%10==0))
            if family=='grid_three':
                self.assertTrue(np.all((rows%3==1).sum(axis=1)==4))
                self.assertTrue(np.all((rows%3==0).sum(axis=1)==1))

    def test_tied_response_matches_exact_dp_on_every_small_population(self):
        game = Game(4,(1,2,3))
        actions = list(iter_allocations(game))
        tables = [Population.from_people([x],game).table() for x in actions]+[uniform_composition(game)]
        for table in tables:
            expected = best_response(table)
            for seed in range(8):
                actual = uniform_best_response(table,seed)
                self.assertEqual(actual.expected_score,expected.expected_score)
                self.assertEqual(actual.optimum_count,expected.optimum_count)
                self.assertEqual(table.expected_score(actual.allocation),expected.expected_score)

    def test_uniform_tie_sampling_does_not_collapse_to_one_lexicographic_action(self):
        game = Game(1,(1,1,1,1,1))
        table = uniform_composition(game)
        counts = Counter(uniform_best_response(table,seed).allocation for seed in range(5000))
        self.assertEqual(len(counts),5)
        self.assertTrue(all(850<count<1150 for count in counts.values()))

    def test_mutation_legality_and_fixed_seed(self):
        for action in [(100,0,0,0,0),(0,0,0,0,100),(20,20,20,20,20)]:
            for seed in range(100):
                x = mutate(action,seed)
                self.assertEqual(sum(x),100)
                self.assertGreaterEqual(min(x),0)
                self.assertEqual(x,mutate(action,seed))
                self.assertLessEqual(sum(abs(a-b) for a,b in zip(x,action)),32)

    def test_path_history_resume_and_memory_mass(self):
        game = Game(6,(1,2,3))
        seed = Population.from_people([(2,2,2)],game)
        environments = [(1,[uniform_composition(game)],[seed.table()])]
        cfg = {'waves':4,'responses_per_wave':9,'mutation_probability':.5,'mutation_steps':2,
            'mutation_max_transfer':2,'strategic_share_percentages':[0,50,100],
            'belief_concentrations':[.05,.5,2]}
        with TemporaryDirectory() as temp:
            for policy in ['latest_only','retain_history']:
                a,b = Path(temp)/f'{policy}-full',Path(temp)/f'{policy}-split'
                full = run_path(a,policy,seed,environments,cfg,100)
                run_path(b,policy,seed,environments,cfg,100,max_wave=2)
                resumed = run_path(b,policy,seed,environments,cfg,100)
                self.assertEqual(full,resumed)
                previous = seed.total_mass
                for record in full:
                    expected = previous if policy=='retain_history' else (
                        seed.total_mass if record['wave']==1 else full[record['wave']-2]['generation']['entries'])
                    if policy=='latest_only' and record['wave']>1:
                        expected = sum(m for x,m in expected)
                    self.assertEqual(record['memory_mass_before'],expected)
                    generation = Population.from_dict(record['generation'])
                    archive = Population.from_dict(record['archive'])
                    self.assertEqual(archive.total_mass,previous+generation.total_mass)
                    previous = archive.total_mass
                state = json.loads((b/'checkpoint.json').read_text())
                state['archive']['entries'][0][1] += 1
                (b/'checkpoint.json').write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    run_path(b,policy,seed,environments,cfg,100)

    def test_checkpoint_rejects_settings_change(self):
        game = Game(3,(1,2))
        population = Population.from_people([(1,2)],game)
        cfg = {'waves':2,'responses_per_wave':3,'mutation_probability':0,'mutation_steps':1,
            'mutation_max_transfer':1,'strategic_share_percentages':[100],
            'belief_concentrations':[1]}
        environment = [(1,[population.table()],[])]
        with TemporaryDirectory() as temp:
            run_path(temp,'retain_history',population,environment,cfg,10,max_wave=1)
            changed = {**cfg,'responses_per_wave':4}
            with self.assertRaises(ValueError):
                run_path(temp,'retain_history',population,environment,changed,10)


if __name__=='__main__':
    unittest.main()
