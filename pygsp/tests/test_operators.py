# -*- coding: utf-8 -*-

"""
Test suite for the operators module of the pygsp package.

"""

import unittest

import numpy as np

from pygsp import graphs, operators


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.G = graphs.Logo()
        cls.G.compute_fourier_basis()

        rs = np.random.RandomState(42)
        cls.signal = rs.uniform(size=cls.G.N)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_difference(self):
        for lap_type in ['combinatorial', 'normalized']:
            G = graphs.Logo(lap_type=lap_type)
            s_grad = operators.grad(G, self.signal)
            Ls = operators.div(G, s_grad)
            np.testing.assert_allclose(Ls, G.L.dot(self.signal))

    def test_translate(self):
        self.assertRaises(NotImplementedError, operators.translate,
                          self.G, self.signal, 42)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
