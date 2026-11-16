import unittest
from pathlib import Path
import numpy as np
from game2_lab.cross_arena import (load_catalog, native_module, draw_population,
    uncompressed_groups, exact_sample_sums, summary, fixed_pool, read)
from game2_lab.core import focal_samples


ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'data/parry-replay'


class CrossArenaTests(unittest.TestCase):
    def test_frozen_catalog_and_field_order(self):
        fit,p,fields=load_catalog(VENDOR)
        self.assertEqual(len(p),40)
        self.assertEqual(sum(len(x)for x,w in p.values()),218922)
        self.assertEqual(len(fields),17)
        self.assertEqual(sum(r['strict']for r in fields),9)
        self.assertEqual(list(p),fit['catalog'])

    def test_family_first_sampling_preserves_joint_allocations(self):
        p={'a':(np.array([[100,0,0,0,0],[0,100,0,0,0]],np.uint8),np.array([1.,3.])),
           'b':(np.array([[0,0,0,0,100]],np.uint8),np.ones(1))}
        rows=draw_population(p,[.6,.4],100,67)
        rng=np.random.default_rng(67)
        f=rng.choice(2,100,p=[.6,.4]);expected=np.empty((100,5),np.uint8)
        for j,(x,w)in enumerate(p.values()):
            ids=np.where(f==j)[0]
            expected[ids]=x[rng.choice(len(x),len(ids),p=w/w.sum())]
        np.testing.assert_array_equal(rows,expected)
        self.assertTrue(np.all(rows.sum(axis=1)==100))

    def test_native_and_foundation_group_cells_and_compression(self):
        native=native_module(VENDOR)
        rng=np.random.default_rng(314)
        rows=rng.multinomial(100,[.1,.2,.25,.3,.15],size=(1024,3)).astype(np.uint8)
        rows[:4]=0
        rows[:4,:,4]=100
        xs=[[0,36,0,64,0],[15,18,49,12,6],[0,0,0,0,100],[20]*5]
        g=uncompressed_groups(rows)
        sums=exact_sample_sums(native,xs,native.compress_groups(rows),len(rows))
        for j,x in enumerate(xs):
            pay,raw,rank=native.outcomes(x,g)
            direct=focal_samples(x,rows)
            np.testing.assert_array_equal(pay*4,direct['payout4'])
            np.testing.assert_array_equal(np.rint(raw*12),direct['castle12'])
            self.assertEqual(int(sums[j]),int(direct['payout4'].sum()))

    def test_selection_rules_and_exact_tie_order(self):
        fields=[{'strict':True},{'strict':False}]
        x=[[1,0,0,0,99],[0,1,0,0,99],[50,50,0,0,0]]
        r=summary([[5,4],[4,5],[3,9]],fields,x)
        self.assertEqual(r['all_17']['maximin_index'],1)
        self.assertEqual(r['strict_9']['maximin_index'],0)
        # A hypothetical best in each column determines regret, not a mean.
        self.assertEqual(r['all_17']['minimax_regret_index'],2)

    def test_pool_union_keeps_provenance_and_deduplicates(self):
        pool=fixed_pool({'pool':{'h':{'allocation':[0,36,0,64,0]}}},
            {'candidates':[{'allocation':[0,36,0,64,0]}]},
            {'pool':[[0,0,0,0,100]],'memory_actions':[[0,36,0,64,0]],'global_best_response':[0,0,0,0,100]})
        self.assertEqual(len(pool),2)
        self.assertEqual(len(pool[0]['sources']),3)
        self.assertEqual(len(pool[1]['sources']),2)


if __name__=='__main__':unittest.main()
