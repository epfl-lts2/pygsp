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
        cls._G.compute_fourier_basis()
        cls._rs = np.random.RandomState(42)
        cls._signal = cls._rs.uniform(size=cls._G.N)

    @classmethod
    def tearDownClass(cls):
        pass

    def _generate_coefficients(self, N, Nf, vertex_delta=83):
        S = np.zeros((N*Nf, Nf))
        S[vertex_delta] = 1
        for i in range(Nf):
            S[vertex_delta + i * self._G.N, i] = 1
        return S

    def _test_methods(self, f, tight):
        self.assertIs(f.G, self._G)

        f.evaluate(self._G.e)

        A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        if tight:
            np.testing.assert_allclose(A, B)
        else:
            assert B - A > 0.01

        # Analysis.
        s2 = f.filter(self._signal, method='exact')
        s3 = f.filter(self._signal, method='chebyshev', order=100)

        # Synthesis.
        s4 = f.filter(s2, method='exact')
        s5 = f.filter(s3, method='chebyshev', order=100)

        if f.Nf < 100:
            # Chebyshev should be close to exact.
            # TODO: does not pass for Gabor.
            np.testing.assert_allclose(s2, s3, rtol=0.1, atol=0.01)
            np.testing.assert_allclose(s4, s5, rtol=0.1, atol=0.01)

        if tight:
            # Tight frames should not loose information.
            np.testing.assert_allclose(s4, A * self._signal)
            assert np.linalg.norm(s5 - A * self._signal) < 0.1

        self.assertRaises(ValueError, f.filter, s2, method='lanczos')

        if f.Nf < 10:
            # Computing the frame is an alternative way to filter.
            # Though it is memory intensive.
            F = f.compute_frame(method='exact')
            F = F.reshape(self._G.N, -1)
            s = F.T.dot(self._signal).reshape(self._G.N, -1).squeeze()
            np.testing.assert_allclose(s, s2)

            F = f.compute_frame(method='chebyshev', order=100)
            F = F.reshape(self._G.N, -1)
            s = F.T.dot(self._signal).reshape(self._G.N, -1).squeeze()
            np.testing.assert_allclose(s, s3)

        # TODO: f.can_dual()

    def test_filter(self):
        Nf = 5
        g = filters.MexicanHat(self._G, Nf=Nf)

        s1 = self._rs.uniform(size=self._G.N)
        s2 = s1.reshape((self._G.N, 1))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.analyze(s1)
        assert s3.shape == (self._G.N, Nf)
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rs.uniform(size=(self._G.N, 4))
        s2 = s1.reshape((self._G.N, 4, 1))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.analyze(s1)
        assert s3.shape == (self._G.N, 4, Nf)
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rs.uniform(size=(self._G.N, Nf))
        s2 = s1.reshape((self._G.N, 1, Nf))
        s3 = g.filter(s1)
        s4 = g.filter(s2)
        s5 = g.synthesize(s1)
        assert s3.shape == (self._G.N,)
        np.testing.assert_allclose(s3, s4)
        np.testing.assert_allclose(s3, s5)

        s1 = self._rs.uniform(size=(self._G.N, 10, Nf))
        s3 = g.filter(s1)
        s5 = g.synthesize(s1)
        assert s3.shape == (self._G.N, 10)
        np.testing.assert_allclose(s3, s5)

    def test_localize(self):
        G = graphs.Grid2d(20)
        G.compute_fourier_basis()
        g = filters.Heat(G, 100)

        # Localize signal at node by filtering Kronecker delta.
        NODE = 10
        s1 = g.localize(NODE, method='exact')

        # Should be equal to a row / column of the filtering operator.
        gL = G.U.dot(np.diag(g.evaluate(G.e)[0]).dot(G.U.T))
        s2 = np.sqrt(G.N) * gL[NODE, :]
        np.testing.assert_allclose(s1, s2)

        # That is actually a row / column of the analysis operator.
        F = g.compute_frame(method='exact')
        np.testing.assert_allclose(F, gL)

    def test_custom_filter(self):
        def kernel(x):
            return x / (1. + x)
        f = filters.Filter(self._G, kernels=kernel)
        self.assertEqual(f.Nf, 1)
        self.assertIs(f._kernels[0], kernel)
        self._test_methods(f, tight=False)

    def test_abspline(self):
        f = filters.Abspline(self._G, Nf=4)
        self._test_methods(f, tight=False)

    def test_gabor(self):
        f = filters.Gabor(self._G, lambda x: x / (1. + x))
        self._test_methods(f, tight=False)

    def test_halfcosine(self):
        f = filters.HalfCosine(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_itersine(self):
        f = filters.Itersine(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_mexicanhat(self):
        f = filters.MexicanHat(self._G, Nf=5, normalize=False)
        self._test_methods(f, tight=False)
        f = filters.MexicanHat(self._G, Nf=4, normalize=True)
        self._test_methods(f, tight=False)

    def test_meyer(self):
        f = filters.Meyer(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_simpletf(self):
        f = filters.SimpleTight(self._G, Nf=4)
        self._test_methods(f, tight=True)

    def test_regular(self):
        f = filters.Regular(self._G)
        self._test_methods(f, tight=True)
        f = filters.Regular(self._G, d=5)
        self._test_methods(f, tight=True)
        f = filters.Regular(self._G, d=0)
        self._test_methods(f, tight=True)

    def test_held(self):
        f = filters.Held(self._G)
        self._test_methods(f, tight=True)
        f = filters.Held(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_simoncelli(self):
        f = filters.Simoncelli(self._G)
        self._test_methods(f, tight=True)
        f = filters.Simoncelli(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_papadakis(self):
        f = filters.Papadakis(self._G)
        self._test_methods(f, tight=True)
        f = filters.Papadakis(self._G, a=0.25)
        self._test_methods(f, tight=True)

    def test_heat(self):
        f = filters.Heat(self._G, normalize=False, tau=10)
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=False, tau=np.array([5, 10]))
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=True, tau=10)
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)), 1)
        self._test_methods(f, tight=False)
        f = filters.Heat(self._G, normalize=True, tau=[5, 10])
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[0]), 1)
        np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[1]), 1)
        self._test_methods(f, tight=False)

    def test_expwin(self):
        f = filters.Expwin(self._G)
        self._test_methods(f, tight=False)

    def test_approximations(self):
        r"""
        Test that the different methods for filter analysis, i.e. 'exact',
        'cheby', and 'lanczos', produce the same output.
        """
        # TODO: done in _test_methods.

        f = filters.Heat(self._G)
        c_exact = f.filter(self._signal, method='exact')
        c_cheby = f.filter(self._signal, method='chebyshev')

        np.testing.assert_allclose(c_exact, c_cheby)
        self.assertRaises(ValueError, f.filter, self._signal, method='lanczos')


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
