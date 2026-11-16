from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from blotto_lab.core import Population,mix_tables
from blotto_lab.historical import validate_rows
from blotto_lab.human import learning_path,score_matrices
from blotto_lab.merged import coupled_path,fresh_definitions,mixture_weights,shifted_rows
from blotto_lab.multiwave import simple_json


class MergedStudyTests(unittest.TestCase):
    def test_multiaffine_box_is_exact_convex_combination_of_corners(self):
        components=[Population.from_people([x]).table() for x in [(20,20,20,20,20),(1,1,20,8,70),(3,3,2,41,51),(0,10,30,30,30),(10,10,0,60,20)]]
        parameters=(Fraction(3,10),Fraction(1,5),Fraction(7,10),Fraction(1,10))
        direct=mix_tables(components,mixture_weights(*parameters))
        axes=[(Fraction(0),Fraction(1)),(Fraction(0),Fraction(1,2)),(Fraction(0),Fraction(1)),(Fraction(0),Fraction(1,4))]
        corners=[];weights=[]
        for bits in product((0,1),repeat=4):
            values=[axes[i][b] for i,b in enumerate(bits)];mass=Fraction(1)
            for i,b in enumerate(bits):
                t=parameters[i]/axes[i][1];mass*=t if b else 1-t
            corners.append(mix_tables(components,mixture_weights(*values)));weights.append(mass)
        expanded=mix_tables(corners,weights)
        for x in [(1,2,26,31,40),(4,19,27,48,2),(20,20,20,20,20)]:self.assertEqual(direct.expected_score(x),expanded.expected_score(x))
        self.assertEqual(sum(mixture_weights(*parameters)),1)

    def test_fresh_scout_and_attention_generators(self):
        validation,audit=fresh_definitions();self.assertEqual((len(validation),len(audit)),(20,15))
        for definition in validation+audit:
            rows,meta=shifted_rows(definition,500,991)
            again,_=shifted_rows(definition,500,991)
            validate_rows(rows);np.testing.assert_array_equal(rows,again)
            self.assertTrue(np.all(rows[:,definition['coalition']]>=1))
            if definition['heavy'] is not None:self.assertTrue(np.all(rows[:,4]>=52))
            if definition['scouts']=='correlated':
                outside=[i for i in range(5) if i not in definition['coalition']]
                self.assertTrue(np.all(rows[:,outside]==rows[:,outside[:1]]))

    def test_single_type_endpoints_match_homogeneous_learning(self):
        actions=[(20,20,20,20,20),(1,1,20,8,70),(3,3,2,41,51)];initial=[.2,.3,.5]
        p2,w2=score_matrices(actions)
        cfg={'learning_steps':10,'invasion_every':100,'invasion_seed_mass':.01,'record_every':5,'quantization_mass':10**12}
        with TemporaryDirectory() as temp:
            for share,matrix in [(0,p2/2),(1,w2/2)]:
                result=coupled_path(Path(temp)/str(share),actions,initial,share,'replicator',.01,cfg,11)
                expected=learning_path(matrix,initial,'replicator',.01,10,5)
                np.testing.assert_allclose(result['final'],expected['final'],atol=1e-14)
                np.testing.assert_allclose(result['average'],expected['average'],atol=1e-14)

    def test_coupled_innovation_resume_and_checkpoint_integrity(self):
        actions=[(20,20,20,20,20),(1,1,20,8,70),(3,3,2,41,51)];initial=np.array([.2,.3,.5]);before=initial.copy()
        cfg={'learning_steps':12,'invasion_every':4,'invasion_seed_mass':.01,'record_every':2,'quantization_mass':10**12}
        with TemporaryDirectory() as temp:
            full=coupled_path(Path(temp)/'full',actions,initial,.75,'replicator',.01,cfg,22)
            coupled_path(Path(temp)/'split',actions,initial,.75,'replicator',.01,cfg,22,max_step=4)
            split=coupled_path(Path(temp)/'split',actions,initial,.75,'replicator',.01,cfg,22)
            self.assertEqual(full['actions'],split['actions'])
            np.testing.assert_array_equal(full['final'],split['final']);np.testing.assert_array_equal(full['average'],split['average'])
            self.assertEqual(simple_json(full['innovations']),simple_json(split['innovations']))
            self.assertGreater(len(full['actions']),len(actions))
            self.assertTrue(all(abs(t['official_population_mean']-7.5)<1e-10 for t in full['trace']))
            self.assertTrue(all(t['fixed_win_share']==.75 for t in full['trace']))
            checkpoint=Path(temp)/'split/checkpoint.json';state=json.loads(checkpoint.read_text());state['point'][0]+=.1;checkpoint.write_text(json.dumps(state))
            with self.assertRaises(ValueError):coupled_path(Path(temp)/'split',actions,initial,.75,'replicator',.01,cfg,22)
        np.testing.assert_array_equal(initial,before)


if __name__=='__main__':unittest.main()
