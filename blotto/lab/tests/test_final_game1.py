from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from final_tools import final_definitions,final_rows,large_mutant,enhanced_path


class FinalGame1Tests(unittest.TestCase):
    def cfg(self):
        return {'learning_steps':12,'invasion_every':3,'invasion_seed_mass':.01,'record_every':3,
                'quantization_mass':10**9,'large_transfer_min':5,'large_transfer_max':25}

    def test_all_new_generator_laws_are_legal_and_reproducible(self):
        for phase in ['challenge','confirmation']:
            defs=final_definitions(phase);self.assertEqual(len(defs),26)
            for i,d in enumerate(defs):
                a=final_rows(d,100,8820+i);b=final_rows(d,100,8820+i)
                np.testing.assert_array_equal(a,b);self.assertTrue(np.all(a>=0));self.assertTrue(np.all(a.sum(axis=1)==100))
                if d['law']=='middle5':self.assertTrue(np.all((a[:,4]>=12)&(a[:,4]<=54)))

    def test_large_transfers_cross_plateaus_without_breaking_budget(self):
        rng=np.random.default_rng(444)
        for x in [(0,0,0,0,100),(3,9,35,47,6),(20,20,20,20,20)]:
            for _ in range(30):
                y,d=large_mutant(x,rng);self.assertEqual(sum(y),100);self.assertGreaterEqual(min(y),0)
                self.assertGreaterEqual(d['amount'],5);self.assertLessEqual(d['amount'],25)
                self.assertEqual(sum(abs(a-b)for a,b in zip(x,y)),2*d['amount'])

    def test_interrupted_innovation_replays_with_identical_random_transfers(self):
        actions=[(20,20,20,20,20),(0,0,50,0,50),(0,0,0,50,50),(3,9,35,47,6)];initial=np.array([.1,.3,.4,.2]);copy=initial.copy()
        with TemporaryDirectory()as temp:
            p=Path(temp);full=enhanced_path(p/'full',actions,initial,.5,'replicator',.01,self.cfg(),991)
            enhanced_path(p/'resumed',actions,initial,.5,'replicator',.01,self.cfg(),991,max_step=5)
            resumed=enhanced_path(p/'resumed',actions,initial,.5,'replicator',.01,self.cfg(),991)
            self.assertEqual(full['actions'],resumed['actions']);self.assertEqual(full['events'],resumed['events'])
            for field in ['point','win','final','average']:np.testing.assert_array_equal(full[field],resumed[field])
            np.testing.assert_array_equal(initial,copy)
            # Checkpoint tampering is detected before state is used.
            path=p/'resumed/checkpoint.json';state=json.loads(path.read_text());state['point'][0]+=.01;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):enhanced_path(p/'resumed',actions,initial,.5,'replicator',.01,self.cfg(),991)

    def test_fixed_type_shares_include_pure_objective_endpoints(self):
        actions=[(20,20,20,20,20),(0,0,50,0,50),(0,0,0,50,50)]
        with TemporaryDirectory()as temp:
            for w in [0,.5,1]:
                result=enhanced_path(Path(temp)/str(w),actions,[.3,.3,.4],w,'payoff_replacement',0,self.cfg(),55)
                np.testing.assert_allclose(result['final'],(1-w)*result['point']+w*result['win'],atol=1e-15,rtol=0)
                self.assertEqual(result['actions'][:len(actions)],actions)
                self.assertTrue(all(abs(t['official_population_mean']-7.5)<1e-12 for t in result['trace']))


if __name__=='__main__':unittest.main()
