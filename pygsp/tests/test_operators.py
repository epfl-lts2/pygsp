# -*- coding: utf-8 -*-

"""
Test suite for the operators module of the pygsp package.

"""

import unittest

import numpy as np
from scipy import sparse

from pygsp import graphs, filters, operators


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.G = graphs.Logo()
        cls.G.compute_fourier_basis()

        cls.rs = np.random.RandomState(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_fourier_transform(self):
        f = self.rs.uniform(size=self.G.N)
        f_hat = operators.gft(self.G, f)
        f_star = operators.igft(self.G, f_hat)
        np.testing.assert_allclose(f, f_star)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
