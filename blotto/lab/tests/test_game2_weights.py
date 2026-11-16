from fractions import Fraction
from itertools import product
import unittest
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import reference_group,exact_evaluation
from game2_lab.weights import (A,FAMILY_IDS,conditional_triplets,family_tensor,copy_coefficients,cubic_value,
 continuous_minimum,polynomial_value,domain_simplices,robust_partition,without_copies,pooled_population,representation_path)


class WeightTests(unittest.TestCase):
    def test_family_tensor_matches_complete_iid_group_enumeration(self):
        support=[A,(1,29,3,64,3),(9,3,0,27,61),(1,1,1,1,96)];x=(3,8,4,44,41)
        means=np.array([float(reference_group([x]+[support[i]for i in ids]).payouts[0])for ids in FAMILY_IDS]);tensor=family_tensor(means)
        for counts in [[1,1,1,1],[2,3,0,1],[0,1,4,0]]:
            pop=Population.from_counts([(a,n)for a,n in zip(support,counts)if n]);expected=exact_evaluation(x,pop).expected_payout
            self.assertAlmostEqual(polynomial_value(tensor,np.array(counts)/sum(counts)),float(expected),places=12)
        q=np.array([.2,.3,.5]);coeff=copy_coefficients(tensor,q)
        for p in [0,.1,.5,.9,1]:self.assertAlmostEqual(cubic_value(coeff,p),polynomial_value(tensor,np.r_[p,(1-p)*q]),places=12)

    def test_continuous_extrema_find_interior_and_flat_minima(self):
        # p^2-p+1/2 has Bernstein coefficients 1/2,1/6,1/6,1/2.
        rows=np.array([[.5,1/6,1/6,.5],[3,3,3,3],[9,0,0,0],[0,3,6,9]])
        values,where=continuous_minimum(rows,1)
        np.testing.assert_allclose(values,[.25,3,0,0],atol=1e-12);self.assertAlmostEqual(where[0],.5)
        value,p=continuous_minimum(rows,.25);self.assertAlmostEqual(value[0],.3125);self.assertAlmostEqual(p[0],.25)
        rng=np.random.default_rng(113)
        for coeff in rng.uniform(0,9,size=(15,4)):
            value,p=continuous_minimum(coeff,.73);grid=cubic_value(coeff,np.linspace(0,.73,20001))
            self.assertLessEqual(float(value),float(grid.min())+1e-11);self.assertAlmostEqual(float(value),float(cubic_value(coeff,p)),places=11)

    def test_domain_partition_volume_and_bernstein_global_bounds(self):
        for cap in [.1,.25,.5,1]:
            volume=sum(abs(np.linalg.det((v[1:,:3]-v[0,:3]).T))/6 for v in domain_simplices(cap))
            self.assertAlmostEqual(volume,(1-(1-cap)**3)/6,places=13)
        # f(w)=(w0-0.2)^2+2; embed its quadratic as a cubic using sum(w)=1.
        tensor=np.empty((4,4,4))
        for i,j,k in product(range(4),repeat=3):
            tensor[i,j,k]=2.04-.4*((i==0)+(j==0)+(k==0))/3+((i==0 and j==0)+(i==0 and k==0)+(j==0 and k==0))/3
        result=robust_partition(tensor,.5,tolerance=.001,max_splits=5000)
        self.assertLessEqual(result['lower'],2);self.assertGreaterEqual(result['upper'],2);self.assertLess(result['gap'],.00101)
        self.assertAlmostEqual(polynomial_value(tensor,result['witness']),result['upper'],places=8)

    def test_conditioned_sampling_is_legal_and_has_requested_copy_counts(self):
        cloud=[A,(2,2,3,4,89),(1,29,3,64,3)]
        for ids in FAMILY_IDS:
            cell={'kind':'family','ids':ids};rows=conditional_triplets(cell,43,112,cloud)
            self.assertTrue(np.all(rows.sum(axis=2)==100));self.assertTrue(np.all((rows[:,:,4]==100).sum(axis=1)==ids.count(0)))
            np.testing.assert_array_equal(rows,conditional_triplets(cell,43,112,cloud))
        for k in range(3):
            cell={'kind':'reference','copies':k,'reference':{'kind':'static','family':'counter_cloud'}}
            rows=conditional_triplets(cell,101,114,cloud);self.assertTrue(np.all((rows[:,:,4]==100).sum(axis=1)==k))

    def test_copy_removal_and_equal_pooling_preserve_probability(self):
        a=(1,29,3,64,3);b=(20,20,20,20,20)
        p=Population.from_counts([(A,2),(a,1)]);q=Population.from_counts([(b,2)])
        rest,mass=without_copies(p);self.assertEqual(mass,Fraction(2,3));self.assertEqual(rest.entries,((a,1),))
        pool=pooled_population([p,q]);prob={x:Fraction(n,pool.total_mass)for x,n in pool.entries}
        self.assertEqual(prob[A],Fraction(1,3));self.assertEqual(prob[a],Fraction(1,6));self.assertEqual(prob[b],Fraction(1,2))

    def test_balanced_replacement_is_duplicate_invariant_but_flat_is_not(self):
        actions=[A,(1,1,1,1,96),(1,29,3,64,3),(20,20,20,20,20),(0,39,0,0,61)]
        cfg={'representation_steps':12,'representation_triplets':128,'representation_mutation':.005};one=np.ones(len(actions));many=one.copy();many[0]=10
        a=representation_path(actions,one,True,cfg,116);b=representation_path(actions,many,True,cfg,116)
        np.testing.assert_array_equal(a['trace'],b['trace']);np.testing.assert_allclose(a['trace'].sum(axis=1),1)
        c=representation_path(actions,one,False,cfg,116);d=representation_path(actions,many,False,cfg,116)
        self.assertGreater(np.abs(c['final']-d['final']).sum(),.01)


if __name__=='__main__':unittest.main()
