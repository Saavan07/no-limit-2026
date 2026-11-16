from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
import numpy as np
from blotto_lab.core import Game,iter_allocations
from game2_lab.core import reference_group,focal_samples
from game2_lab.arena import single_triplet_response,metric_arrays,mutate_action,arena_path,initial_shares,static_triplets


class Game2ArenaTests(unittest.TestCase):
    def cfg(self):
        return {'steps':12,'fitness_triplets':64,'innovation_every':4,'settling_steps':4,'seed_mass':.01,
                'archive_proposals':4,'mutation_proposals':4,'random_proposals':4,'search_triplets':64,'gate_triplets':128,
                'maximum_rank_gates':240,'gate_alpha':.05,'record_every':4,'quantization_mass':10**9}

    def seeds(self):return [(0,0,0,0,100),(1,1,1,1,96),(20,20,20,20,20),(1,29,3,64,3),(2,30,4,60,4),(0,0,100,0,0),(0,0,0,100,0)]

    def test_exact_single_group_oracle_against_complete_small_games(self):
        rng=np.random.default_rng(222)
        for game in [Game(4,(1,2,3)),Game(2,(1,2,3,4,5))]:
            actions=list(iter_allocations(game))
            for _ in range(12):
                opponents=[actions[int(i)]for i in rng.integers(len(actions),size=3)];response=single_triplet_response(opponents,game)
                values=[(reference_group([x]+opponents,game).payouts[0],x)for x in actions];top=max(v for v,x in values)
                self.assertEqual(response['payout'],top);self.assertEqual(response['allocation'],min(x for v,x in values if v==top))

    def test_default_worst_group_caps_all_focal_choices_at_one(self):
        opponents=[(0,0,100,0,0),(0,0,0,100,0),(0,0,0,0,100)]
        response=single_triplet_response(opponents);self.assertEqual(response['payout'],1)
        self.assertEqual(reference_group([(0,0,0,0,100)]+opponents).payouts[0],1)

    def test_three_metric_arrays_match_foundation(self):
        rng=np.random.default_rng(333);triples=rng.multinomial(100,rng.dirichlet(np.ones(5),size=111)).reshape(37,3,5)
        actions=self.seeds();rank,castle,first=metric_arrays(actions,triples,candidate_batch=3,triplet_batch=11)
        for i,x in enumerate(actions):
            ref=focal_samples(x,triples);np.testing.assert_array_equal(rank[i],ref['payout4']);np.testing.assert_array_equal(castle[i],ref['castle12']);np.testing.assert_array_equal(first[i],ref['finish_credits12'][:,0])

    def test_mutations_keep_full_budget_and_can_reach_high_scouts(self):
        rng=np.random.default_rng(444);saw_high=False
        for x in self.seeds():
            for _ in range(30):
                y=mutate_action(x,(40,50,0,0,10),rng,True);self.assertEqual(sum(y),100);self.assertGreaterEqual(min(y),0)
                saw_high|=y[0]>=39 or y[1]>=49
                z=mutate_action(x,y,rng,False);self.assertEqual(sum(z),100);self.assertGreaterEqual(min(z),0)
        self.assertTrue(saw_high)

    def test_path_recovery_has_identical_rng_events_and_settling(self):
        actions=self.seeds();initial=initial_shares(actions,'balanced',2);copy=initial.copy()
        with TemporaryDirectory()as temp:
            root=Path(temp);a=arena_path(root/'a',actions,actions,initial,[1,0,0],'replicator',.25,.005,self.cfg(),555)
            arena_path(root/'b',actions,actions,initial,[1,0,0],'replicator',.25,.005,self.cfg(),555,max_step=5)
            b=arena_path(root/'b',actions,actions,initial,[1,0,0],'replicator',.25,.005,self.cfg(),555)
            self.assertEqual(a['actions'],b['actions']);self.assertEqual(a['waves'],b['waves']);self.assertEqual(a['trace'],b['trace'])
            for field in ['types','final','average','tail_average']:np.testing.assert_array_equal(a[field],b[field])
            self.assertEqual(a['tail_count'],4);self.assertTrue(all(w['step']<=8 for w in a['waves']));np.testing.assert_array_equal(initial,copy)
            path=root/'b/checkpoint.json';state=json.loads(path.read_text());state['types'][0][0]+=.1;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):arena_path(root/'b',actions,actions,initial,[1,0,0],'replicator',.25,.005,self.cfg(),555)

    def test_mixed_objective_shares_remain_fixed(self):
        actions=self.seeds();initial=initial_shares(actions,'heavy_five',3)
        with TemporaryDirectory()as temp:
            result=arena_path(Path(temp),actions,actions,initial,[.6,.2,.2],'payoff_replacement',.25,0,self.cfg(),666)
            np.testing.assert_allclose(result['final'],np.array([.6,.2,.2])@result['types'],rtol=0,atol=1e-15)
            self.assertEqual(result['actions'][:len(actions)],actions)
            self.assertTrue(all(t['type_shares']==[.6,.2,.2]and t['selfplay_group_mean']==3.75 for t in result['trace']))

    def test_new_static_challenges_are_legal_and_reproducible(self):
        for family in ['rotated','wide_abandon','middle_five','counter_cloud']:
            for variant in [0,1]:
                rows=static_triplets(family,variant,20,777,self.seeds());again=static_triplets(family,variant,20,777,self.seeds())
                np.testing.assert_array_equal(rows,again);self.assertTrue(np.all(rows>=0));self.assertTrue(np.all(rows.sum(axis=2)==100))


if __name__=='__main__':unittest.main()
