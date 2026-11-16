from pathlib import Path
from tempfile import TemporaryDirectory
import json,unittest
import numpy as np
from game2_lab.final import prior_weights,residual_rows,core_triplets,pressure_triplets,pressure_rows,final_arena
from game2_lab.arena import initial_shares


class Game2FinalTests(unittest.TestCase):
    def actions(self):return [(0,0,0,0,100),(0,0,0,100,0),(0,0,100,0,0),(0,100,0,0,0),(100,0,0,0,0),(1,1,1,1,96),(0,1,10,86,3),(9,3,0,27,61),(20,20,20,20,20)]
    def cfg(self):return {'steps':12,'innovation_every':4,'settling_steps':4,'fitness_triplets':64,'search_triplets':64,'gate_triplets':128,'maximum_rank_gates':96,'gate_alpha':.05,'history_mass':.25,'mutation':.005,'quantization_mass':10**9,'mutation_proposals':8,'archive_proposals':4,'random_proposals':4,'seed_mass':.003,'record_every':4}

    def test_prior_class_masses_and_duplicate_label_aggregation(self):
        actions=self.actions();shape=prior_weights(actions,'shape');self.assertAlmostEqual(shape[:5].sum(),.2);self.assertAlmostEqual(shape[5],.2);self.assertAlmostEqual(shape[6],.3);self.assertAlmostEqual(shape[7:].sum(),.3)
        for kind in ['shape','fifth']:
            prior=prior_weights(actions,kind);multiplicity=np.arange(1,len(actions)+1);flat=np.concatenate([np.full(n,p/n)for p,n in zip(prior,multiplicity)])
            aggregated=np.array([flat[sum(multiplicity[:i]):sum(multiplicity[:i+1])].sum()for i in range(len(actions))]);np.testing.assert_allclose(aggregated,prior,atol=1e-16)

    def test_fresh_core_rows_are_legal_reproducible_and_copy_conditioned(self):
        for style in ['coalition','rounded','edge','counter_aware']:
            for k in range(4):
                d={'style':style,'mixture':[.4,.3,.3],'copies':k};a=core_triplets(d,121,193,self.actions());b=core_triplets(d,121,193,self.actions());np.testing.assert_array_equal(a,b)
                self.assertTrue(np.all(a.sum(axis=2)==100));self.assertTrue(np.all((a[:,:,4]==100).sum(axis=1)==k))
        rows=residual_rows('coalition',10000,np.random.default_rng(194),[1,0,0],self.actions());self.assertGreater(np.all(rows==[0,0,0,100,0],axis=1).sum(),200)

    def test_pressure_scenarios_mix_people_and_allow_crowding(self):
        for family in ['threshold_four','crowded_four','cheap_overbid','shifted_near']:
            for mass in [.1,.3,.6]:
                d={'family':family,'mass':mass};a=pressure_triplets(d,97,195,self.actions());np.testing.assert_array_equal(a,pressure_triplets(d,97,195,self.actions()));self.assertTrue(np.all(a.sum(axis=2)==100))
            self.assertEqual(pressure_rows(family,0,np.random.default_rng(196),self.actions()).shape,(0,5))

    def test_new_arena_interruption_recovery_and_integrity(self):
        for prior in ['fifth','shape']:
            for rule in ['replicator','prior_replacement']:
                with TemporaryDirectory()as t:
                    p=Path(t);a=final_arena(p/'full',self.actions(),self.actions(),self.cfg(),rule,prior,'balanced',197)
                    final_arena(p/'resume',self.actions(),self.actions(),self.cfg(),rule,prior,'balanced',197,max_step=5)
                    b=final_arena(p/'resume',self.actions(),self.actions(),self.cfg(),rule,prior,'balanced',197)
                    for key in ['actions','trace','waves','input_digest']:self.assertEqual(a[key],b[key])
                    for key in ['final','average','tail']:np.testing.assert_array_equal(a[key],b[key])
                    self.assertEqual(a['tail_count'],4);self.assertTrue(all(w['step']<=8 for w in a['waves']));self.assertTrue(all(x['selfplay_group_mean']==3.75 for x in a['trace']))
                    f=p/'resume/checkpoint.json';r=json.loads(f.read_text());r['q'][0]+=.1;f.write_text(json.dumps(r))
                    with self.assertRaises(ValueError):final_arena(p/'resume',self.actions(),self.actions(),self.cfg(),rule,prior,'balanced',197)


    def test_frozen_selection_uses_regret_and_reports_distinct_alternatives(self):
        from game2_lab.final import selection_policy,evaluation_weights
        cells=['k0','k1','k2','all_A','pressure','path__average','path__tail','path__final']
        design={'core_references':{'reference':['k0','k1','k2','all_A']},'pressure':{'pressure':{}},'arena_paths':[{'name':'path'}]}
        cfg={'evaluation_triplets':2,'copy_points':[0,.1,.25,.5]};candidates={str(i):{'allocation':x}for i,x in enumerate(self.actions()[:3])}
        values=np.array([[9,9,9,9,1,1,1,0],[5,5,5,5,5,5,5,0],[4,4,4,4,6,6,6,0]])*8
        result=selection_policy(values,cells,candidates,design,cfg)
        self.assertEqual(result['selected_candidate_id'],'1');self.assertEqual(result['alternatives']['equal_block_mean'],'2');self.assertEqual(result['alternatives']['low_copy_core_mean'],'0')
        self.assertTrue(all(sum(w.values())==1 for w in evaluation_weights(design,cfg).values()))


if __name__=='__main__':unittest.main()
