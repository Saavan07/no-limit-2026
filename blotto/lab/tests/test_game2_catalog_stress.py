import itertools
import unittest
import numpy as np
from game2_lab.catalog_stress import (compositions,target_family,make_catalog,CubicCDF,
    choose_diverse,split_actions,paired_moments,cubic_minimum,empirical_bernstein_error)


class CatalogStressTests(unittest.TestCase):
    def test_target_generation_preserves_budget_and_active_minimum(self):
        for size in [2,3]:
            for active in itertools.combinations(range(5),size):
                for kind in ['zero_scout_targets','broad_scout_targets','high_scout_targets','uniform_active_targets','equal_active_targets','value_active_targets','corner_active_targets']:
                    x,w=target_family(active,kind,24,91)
                    self.assertTrue(np.all(x.sum(axis=1)==100))
                    self.assertTrue(np.all(x[:,active]>=8))
                    self.assertAlmostEqual(w.sum(),1.)
        self.assertEqual(len(compositions(100,2)),85)

    def test_catalog_perturbation_preserves_named_atoms(self):
        base={'exact_allin5':(np.array([[0,0,0,0,100]],np.uint8),np.ones(1)),
              'balanced':(np.array([[20]*5],np.uint8),np.ones(1)),
              'individual_target_24':(np.array([[0,36,0,64,0]],np.uint8),np.ones(1))}
        for kind in ['nonatom_transfer1','nonatom_transfer5','c4_threshold_upweight','c24_threshold_family']:
            p=make_catalog(base,kind,64,100)
            np.testing.assert_array_equal(p['exact_allin5'][0],base['exact_allin5'][0])
            self.assertTrue(all(np.all(x.sum(axis=1)==100)for x,w in p.values()))

    def test_cubic_payoff_and_cdf_gradients_against_finite_differences(self):
        rng=np.random.default_rng(7);k=3;families=np.repeat(np.arange(k),4)
        ids=np.array(list(itertools.combinations_with_replacement(range(k),3)))
        coeff=rng.uniform(0,9,(len(families),len(ids)))
        thresholds=np.linspace(0,9,13);observed=np.linspace(0,1,13)
        model=CubicCDF(coeff,ids,families,thresholds,observed,np.ones(13)/13,.3)
        w=np.array([.2,.3,.5]);u,du=model.scores_and_gradient(w)
        mult=np.array([1 if len(set(t))==1 else 3 if len(set(t))==2 else 6 for t in ids])
        np.testing.assert_allclose(u,coeff@(mult*np.prod(w[ids],axis=1)),rtol=0,atol=1e-12)
        loss,grad,*_=model.stats(w)
        for j in range(k):
            delta=np.zeros(k);delta[j]=1e-6
            up=model.scores_and_gradient(w+delta)[0];um=model.scores_and_gradient(w-delta)[0]
            np.testing.assert_allclose((up-um)/2e-6,du[:,j],rtol=1e-7,atol=1e-7)
            numeric=(model.stats(w+delta)[0]-model.stats(w-delta)[0])/2e-6
            self.assertAlmostEqual(numeric,grad[j],places=6)

    def test_exact_paired_moments_retain_ties_and_large_counts(self):
        q=np.array([[0,36,15,26],[36,0,15,6],[15,15,15,15]],np.uint8)
        counts=np.array([100003,7,13,16],np.uint64)
        sums,cross=paired_moments(q*3,counts)
        for i in range(3):
            self.assertEqual(int(sums[i]),sum(int(q[i,g])*int(counts[g])for g in range(4)))
            for j in range(3):self.assertEqual(int(cross[i,j]),sum(int(q[i,g])*int(q[j,g])*int(counts[g])for g in range(4)))

    def test_complete_split_family_and_model_diversity(self):
        x=split_actions();self.assertEqual(x.shape,(101,5));self.assertEqual(len(np.unique(x,axis=0)),101)
        self.assertTrue(np.all(x.sum(axis=1)==100))
        rows=[{'run':0,'loss':1,'weights':[1,0]},{'run':1,'loss':2,'weights':[1,0]},{'run':2,'loss':3,'weights':[0,1]}]
        self.assertEqual([r['run']for r in choose_diverse(rows,2)],[0,2])

    def test_cubic_interval_minimum_includes_interior_and_linear_cases(self):
        coeff=np.array([[1,0,0,1],[5,5,5,5],[0,1,2,3]],dtype=float)
        np.testing.assert_allclose(cubic_minimum(coeff),[.25,5,0],atol=1e-12)
        np.testing.assert_allclose(cubic_minimum(coeff,.1),[.73,5,0],atol=1e-12)

    def test_empirical_bound_scales_range_and_accounts_for_multiplicity(self):
        a=empirical_bernstein_error(.2,1000,10,.05,9)
        b=empirical_bernstein_error(.8,1000,10,.05,18)
        self.assertAlmostEqual(2*a,b)
        self.assertGreater(empirical_bernstein_error(.2,1000,100,.05,9),a)


if __name__=='__main__':unittest.main()
