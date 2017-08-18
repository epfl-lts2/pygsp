#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the filters module of the pygsp package.

"""

import unittest

import numpy as np

from pygsp import graphs, filters


class FunctionsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = graphs.Logo()

    @classmethod
    def tearDownClass(cls):
        pass

    def _fu(x):
        return x / (1. + x)

    def test_default_filters(self):
        filters.Filter(self._G)
        filters.Filter(self._G, filters=self._fu)

    def test_abspline(self):
        filters.Abspline(self._G, Nf=4)

    def test_expwin(self):
        filters.Expwin(self._G)

    def test_gabor(self):
        filters.Gabor(self._G, self._fu)

    def test_halfcosine(self):
        filters.HalfCosine(self._G, Nf=4)

    def test_heat(self):
        filters.Heat(self._G)

    def test_held(self):
        filters.Held(self._G)
        filters.Held(self._G, a=0.25)

    def test_itersine(self):
        filters.Itersine(self._G, Nf=4)

    def test_mexicanhat(self):
        filters.MexicanHat(self._G, Nf=5)
        filters.MexicanHat(self._G, Nf=4)

    def test_meyer(self):
        filters.Meyer(self._G, Nf=4)

    def test_papadakis(self):
        filters.Papadakis(self._G)
        filters.Papadakis(self._G, a=0.25)

    def test_regular(self):
        filters.Regular(self._G)
        filters.Regular(self._G, d=5)
        filters.Regular(self._G, d=0)

    def test_simoncelli(self):
        filters.Simoncelli(self._G)
        filters.Simoncelli(self._G, a=0.25)

    def test_simpletf(self):
        filters.SimpleTf(self._G, Nf=4)

    # Warped translates are not implemented yet
    def test_warpedtranslates(self):
        pass
        # gw = filters.warpedtranslates(G, g))

    def test_approximations(self):
        r"""
        Test that the different methods for filter analysis, i.e. 'exact',
        'cheby', and 'lanczos', produce the same output.
        """

        # Signal is a Kronecker delta at node 83.
        s = np.zeros(self._G.N)
        s[83] = 1

        g = filters.Heat(self._G)
        c_exact = g.analysis(s, method='exact')
        c_cheby = g.analysis(s, method='cheby')

        assert np.allclose(c_exact, c_cheby)
        self.assertRaises(NotImplementedError, g.analysis, s, method='lanczos')


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)
