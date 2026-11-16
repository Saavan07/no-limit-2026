import unittest
import itertools
import numpy as np
from blotto_lab.core import Game,iter_allocations
from game2_lab.core import focal_samples
from game2_lab.cross_arena import native_module,uncompressed_groups
from game2_lab.fullscan import allocations,all_sample_sums,direct_sums,allocation_indices,direct_matrix4
from pathlib import Path


class FullScanTests(unittest.TestCase):
    def test_all_actions_on_reduced_games_against_independent_foundation(self):
        rng=np.random.default_rng(734)
        for budget in [0,1,2,4,8]:
            game=Game(budget=budget);rows=rng.multinomial(budget,[.1,.2,.2,.3,.2],size=(80,3)).astype(np.uint8)
            rows[:5]=0
            for j in range(5):rows[j,:,j]=budget
            groups=uncompressed_groups(rows);counts=np.ones(len(rows),np.int64);counts[:4]=[2,3,4,7]
            xs=allocations(budget);np.testing.assert_array_equal(xs,np.array(list(iter_allocations(game))))
            result,_=all_sample_sums(groups,counts,budget)
            for j,x in enumerate(xs):
                expected=focal_samples(x.tolist(),rows,game)['payout4']@counts
                self.assertEqual(int(result[j]),int(expected))
            np.testing.assert_array_equal(direct_sums(xs,groups,counts,1),result)
            np.testing.assert_array_equal(direct_sums(xs,groups,counts,4),result)
            matrix=direct_matrix4(xs,groups,4)
            for j,x in enumerate(xs):np.testing.assert_array_equal(matrix[j],focal_samples(x.tolist(),rows,game)['payout4'])

    def test_full_budget_sparse_and_zero_tie_groups(self):
        # One complete 4.6-million-action transform tests full-size packing and indexing.
        rows=np.array([[[0,0,0,0,100]]*3,[[0,36,0,64,0]]*3,
            [[100,0,0,0,0],[0,0,100,0,0],[0,0,0,0,100]],
            [[20]*5]*3],np.uint8)
        groups=uncompressed_groups(rows);counts=np.array([3,5,7,11],np.int64)
        values,_=all_sample_sums(groups,counts)
        xs=allocations();rng=np.random.default_rng(17);ids=np.unique(np.r_[0,len(xs)-1,rng.choice(len(xs),256,replace=False)])
        for j in ids:self.assertEqual(int(values[j]),int(focal_samples(xs[j].tolist(),rows)['payout4']@counts))
        self.assertTrue(np.all((values>=0)&(values<=36*counts.sum())))

    def test_validation_rejects_unsafe_counts(self):
        rows=np.array([[[20]*5]*3],np.uint8);g=uncompressed_groups(rows)
        for bad in [[0],[-1],[.5]]:
            with self.assertRaises(ValueError):all_sample_sums(g,bad)

    def test_exhaustive_profiles_in_small_games(self):
        for budget in [1,2]:
            x=allocations(budget);game=Game(budget=budget)
            rows=x[np.array(list(itertools.product(range(len(x)),repeat=3)))];g=uncompressed_groups(rows)
            values,_=all_sample_sums(g,np.ones(len(rows),np.int64),budget)
            for i,a in enumerate(x):self.assertEqual(int(values[i]),int(focal_samples(a.tolist(),rows,game)['payout4'].sum()))

    def test_inverse_indices_cover_the_full_action_space(self):
        for budget in [0,1,4,100]:
            x=allocations(budget);np.testing.assert_array_equal(allocation_indices(x,budget),np.arange(len(x)))


if __name__=='__main__':unittest.main()
