import unittest
import numpy as np

import sys
sys.path.append('..')
from transform import standardize, inverse_standardize

class TestTransform(unittest.TestCase):

    mean = 2
    var  = 4
    a = np.array([1, 2, 3, 4])
    b = np.array([-0.5, 0., 0.5, 1.])

    def test_standarize(self):
        b = standardize(self.a, self.mean, self.var)
        assert np.all(b == self.b)

    def test_inverse_standardize(self):
        a = inverse_standardize(self.b, self.mean, self.var)
        assert np.all(a == self.a)

if __name__ == '__main__':
    unittest.main()