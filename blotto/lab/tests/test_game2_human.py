from fractions import Fraction
import unittest
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import focal_samples,reference_group
from game2_lab.human import payoff_arrays,anchor_support,triples_for_support,triple_weights,counts_for_mix,fresh_triplets


class Game2HumanTests(unittest.TestCase):
    def test_reused_opponent_scorer_matches_foundation_for_all_cells(self):
        rng=np.random.default_rng(551);c=rng.multinomial(100,rng.dirichlet(np.ones(5),size=17))
        t=rng.multinomial(100,rng.dirichlet(np.ones(5),size=(91*3))).reshape(91,3,5)
        t[:2]=[[(0,0,0,0,100)]*3,[(20,20,20,20,20)]*3]
        p,s=payoff_arrays([tuple(map(int,x))for x in c],t,candidate_batch=5,triplet_batch=17)
        for j,x in enumerate(c):
            expected=focal_samples(tuple(map(int,x)),t)
            np.testing.assert_array_equal(p[j],expected['payout4']);np.testing.assert_array_equal(s[j],expected['castle12'])

    def test_exact_mixture_weights_and_selfplay(self):
        support=anchor_support();ids,t=triples_for_support(support);counts=counts_for_mix([Fraction(1,4)]*4,200)
        weights=triple_weights(ids,counts);p,s=payoff_arrays(support,t)
        expected=sum(int(n)*int(v)for n,v in zip(counts,p.astype(np.int64)@weights))
        self.assertEqual(Fraction(expected,4*sum(counts)**4),Fraction(15,4))
        self.assertEqual(int(weights.sum()),200**3)
        with self.assertRaises(ValueError):counts_for_mix([0,0,0,0])

    def test_clone_payout_corrections_and_profitable_deviation(self):
        a=(0,0,0,0,100)
        cases=[([a,a,(50,50,0,0,0),(0,0,50,50,0)],1),
               ([a,a,(0,25,25,50,0),(100,0,0,0,0)],3),
               ([a,a,a,(100,0,0,0,0)],5),([a,a,a,(0,100,0,0,0)],2)]
        for rows,expected in cases:self.assertEqual(reference_group(rows).payouts[0],expected)
        self.assertEqual(reference_group([(1,1,1,1,96),a,a,a]).payouts[0],9)

    def test_fresh_families_are_legal_and_preserve_exact_copy_definition(self):
        for style in ['noisy','rounded','contrarian']:
            a,types=fresh_triplets([Fraction(1,4)]*4,style,200,821)
            b,_=fresh_triplets([Fraction(1,4)]*4,style,200,821);np.testing.assert_array_equal(a,b)
            self.assertTrue(np.all(a>=0));self.assertTrue(np.all(a.sum(axis=2)==100))
            self.assertTrue(np.all(a[:,:,4][types==0]==100));self.assertTrue(np.all(a[:,:,4][types!=0]<100))
            only,_=fresh_triplets([1,0,0,0],style,5,2);self.assertTrue(np.all(only[:,:,4]==100))

    def test_broad_allocation_can_outperform_both_proposed_families(self):
        spread=(10,20,40,20,10);a=(0,0,0,0,100);ignore=(2,5,35,58,0)
        result=reference_group([spread,a,a,ignore])
        self.assertEqual(result.castle_points,(Fraction(6),Fraction(5,2),Fraction(5,2),Fraction(4)))
        self.assertEqual(result.payouts,(Fraction(9),Fraction(1),Fraction(1),Fraction(4)))


if __name__=='__main__':unittest.main()
