# -*- coding: utf-8 -*-

"""
Test suite for the filters module of the pygsp package.

"""

import unittest

import numpy as np

from pygsp import graphs, filters


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = graphs.Logo()
        rs = np.random.RandomState(42)
        cls._signal = rs.uniform(size=cls._G.N)

    @classmethod
    def tearDownClass(cls):
        pass

    def _generate_coefficients(self, N, Nf, vertex_delta=83):
        S = np.zeros((N*Nf, Nf))
        S[vertex_delta] = 1
        for i in range(Nf):
            S[vertex_delta + i * self._G.N, i] = 1
        return S

    def _test_synthesis(self, f):
        if 1 < f.Nf < 10:
            S = self._generate_coefficients(f.G.N, f.Nf)
            f.synthesis(S, method='chebyshev')
            f.synthesis(S, method='exact')
            self.assertRaises(NotImplementedError, f.synthesis, S,
                              method='lanczos')

    def _test_methods(self, f):
        self.assertIs(f.G, self._G)

        f.analysis(self._signal, method='exact')
        f.analysis(self._signal, method='chebyshev')
        # TODO np.testing.assert_allclose(c_exact, c_cheby)
        self.assertRaises(NotImplementedError, f.analysis,
                          self._signal, method='lanczos')

        self._test_synthesis(f)
        f.evaluate(np.ones(10))

        f.filterbank_bounds()
        # f.filterbank_matrix()  TODO: too much memory

        # TODO: f.can_dual()

        self.assertRaises(NotImplementedError, f.approx, 0, 0)
        self.assertRaises(NotImplementedError, f.inverse, 0)
        self.assertRaises(NotImplementedError, f.tighten)

    def test_custom_filter(self):
        def _filter(x):
            return x / (1. + x)
        f = filters.Filter(self._G, filters=_filter)
        self.assertIs(f.g[0], _filter)
        self._test_methods(f)

    def test_abspline(self):
        f = filters.Abspline(self._G, Nf=4)
        self._test_methods(f)

    def test_gabor(self):
        f = filters.Gabor(self._G, lambda x: x / (1. + x))
        self._test_methods(f)

    def test_halfcosine(self):
        f = filters.HalfCosine(self._G, Nf=4)
        self._test_methods(f)

    def test_itersine(self):
        f = filters.Itersine(self._G, Nf=4)
        self._test_methods(f)

    def test_mexicanhat(self):
        f = filters.MexicanHat(self._G, Nf=5, normalize=False)
        self._test_methods(f)
        f = filters.MexicanHat(self._G, Nf=4, normalize=True)
        self._test_methods(f)

    def test_meyer(self):
        f = filters.Meyer(self._G, Nf=4)
        self._test_methods(f)

    def test_simpletf(self):
        f = filters.SimpleTf(self._G, Nf=4)
        self._test_methods(f)

    def test_warpedtranslates(self):
        self.assertRaises(NotImplementedError,
                          filters.WarpedTranslates, self._G)
        pass

    def test_regular(self):
        f = filters.Regular(self._G)
        self._test_methods(f)
        f = filters.Regular(self._G, d=5)
        self._test_methods(f)
        f = filters.Regular(self._G, d=0)
        self._test_methods(f)

    def test_held(self):
        f = filters.Held(self._G)
        self._test_methods(f)
        f = filters.Held(self._G, a=0.25)
        self._test_methods(f)

    def test_simoncelli(self):
        f = filters.Simoncelli(self._G)
        self._test_methods(f)
        f = filters.Simoncelli(self._G, a=0.25)
        self._test_methods(f)

    def test_papadakis(self):
        f = filters.Papadakis(self._G)
        self._test_methods(f)
        f = filters.Papadakis(self._G, a=0.25)
        self._test_methods(f)

    def test_heat(self):
        f = filters.Heat(self._G, normalize=False, tau=10)
        self._test_methods(f)
        f = filters.Heat(self._G, normalize=False, tau=[5, 10])
        self._test_methods(f)
        f = filters.Heat(self._G, normalize=True, tau=10)
        self._test_methods(f)
        f = filters.Heat(self._G, normalize=True, tau=[5, 10])
        self._test_methods(f)

    def test_expwin(self):
        f = filters.Expwin(self._G)
        self._test_methods(f)

    def test_approximations(self):
        r"""
        Test that the different methods for filter analysis, i.e. 'exact',
        'cheby', and 'lanczos', produce the same output.
        """
        # TODO: synthesis

        f = filters.Heat(self._G)
        c_exact = f.analysis(self._signal, method='exact')
        c_cheby = f.analysis(self._signal, method='chebyshev')

        np.testing.assert_allclose(c_exact, c_cheby)
        self.assertRaises(NotImplementedError, f.analysis,
                          self._signal, method='lanczos')


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
