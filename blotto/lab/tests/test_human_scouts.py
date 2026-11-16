import unittest
import numpy as np
from blotto_lab.human_scouts import broad_scout_rows
from blotto_lab.historical import validate_rows


class BroadScoutTests(unittest.TestCase):
    def test_legal_reproducible_and_conditioned(self):
        for target in [(2,4),(0,2,3),(0,1,4)]:
            for law in ['uniform','value_multinomial']:
                for heavy in ([None,50] if 4 in target else [None]):
                    rows,meta=broad_scout_rows(target,law,500,123,heavy)
                    again,_=broad_scout_rows(target,law,500,123,heavy)
                    validate_rows(rows);np.testing.assert_array_equal(rows,again)
                    self.assertTrue(np.all(rows[:,target]>=1))
                    outside=[i for i in range(5) if i not in target]
                    self.assertTrue(np.all(rows[:,outside]<=24))
                    if heavy:self.assertTrue(np.all(rows[:,4]>=50))
                    self.assertEqual(meta['accepted'],500)

    def test_uniform_positive_composition_moments_when_outside_zero(self):
        rows,_=broad_scout_rows((2,4),'uniform',30000,567,scout_max=0)
        self.assertAlmostEqual(float(rows[:,2].mean()),50,delta=.6)
        self.assertAlmostEqual(float((rows[:,4]<=24).mean()),24/99,delta=.01)
        with self.assertRaises(ValueError):broad_scout_rows((0,2,3),'uniform',5,123,50)


if __name__=='__main__':unittest.main()
