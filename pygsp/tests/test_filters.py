# -*- coding: utf-8 -*-

"""
Test suite for the filters module of the pygsp package.

"""

import unittest
import sys

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

    def _test_methods(self, f, tight, check=True):
        self.assertIs(f.G, self._G)

        f.evaluate(self._G.e)
        f.evaluate(np.random.uniform(0, 1, size=(4, 6, 3)))

        A, B = f.estimate_frame_bounds(self._G.e)
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

        if check:
            # Chebyshev should be close to exact.
            # Does not pass for Gabor and Rectangular (not smooth).
            np.testing.assert_allclose(s2, s3, rtol=0.1, atol=0.01)
            np.testing.assert_allclose(s4, s5, rtol=0.1, atol=0.01)

        if tight:
            # Tight frames should not loose information.
            np.testing.assert_allclose(s4, A * self._signal)
            assert np.linalg.norm(s5 - A * self._signal) < 0.1

        if f.Nf < 10:
            # Computing the frame is an alternative way to filter.
            # Though it is memory intensive.
            F = f.compute_frame(method='exact')
            s = F.dot(self._signal).reshape(-1, self._G.N).T.squeeze()
            np.testing.assert_allclose(s, s2)

            F = f.compute_frame(method='chebyshev', order=100)
            s = F.dot(self._signal).reshape(-1, self._G.N).T.squeeze()
            np.testing.assert_allclose(s, s3)

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

    def test_frame_bounds(self):
        # Not a frame, it as a null-space.
        g = filters.Rectangular(self._G)
        A, B = g.estimate_frame_bounds()
        self.assertEqual(A, 0)
        self.assertEqual(B, 1)
        # Identity is tight.
        g = filters.Filter(self._G, lambda x: np.full_like(x, 2))
        A, B = g.estimate_frame_bounds()
        self.assertEqual(A, 4)
        self.assertEqual(B, 4)

    def test_frame(self):
        """The frame is a stack of functions of the Laplacian."""
        G = graphs.Sensor(100, seed=42)
        g = filters.Heat(G, tau=[8, 9])
        gL1 = g.compute_frame(method='exact')
        gL2 = g.compute_frame(method='chebyshev', order=30)
        def get_frame(freq_response):
            return G.U.dot(np.diag(freq_response).dot(G.U.T))
        gL = np.concatenate([get_frame(gl) for gl in g.evaluate(G.e)])
        np.testing.assert_allclose(gL1, gL)
        np.testing.assert_allclose(gL2, gL)

    def test_complement(self):
        """Any filter bank becomes tight upon addition of their complement."""
        g = filters.MexicanHat(self._G)
        g += g.complement()
        A, B = g.estimate_frame_bounds()
        np.testing.assert_allclose(A, B)

    def test_inverse(self):
        """The frame is the pseudo-inverse of the original frame."""
        g = filters.Heat(self._G, tau=[2, 3, 4])
        h = g.inverse()
        Ag, Bg = g.estimate_frame_bounds()
        Ah, Bh = h.estimate_frame_bounds()
        np.testing.assert_allclose(Ag * Bh, 1)
        np.testing.assert_allclose(Bg * Ah, 1)
        gL = g.compute_frame(method='exact')
        hL = h.compute_frame(method='exact')
        I = np.identity(self._G.N)
        np.testing.assert_allclose(hL.T.dot(gL), I, atol=1e-10)
        pinv = np.linalg.inv(gL.T.dot(gL)).dot(gL.T)
        np.testing.assert_allclose(pinv, hL.T, atol=1e-10)
        # The reconstruction is exact for any frame (lower bound A > 0).
        y = g.filter(self._signal, method='exact')
        z = h.filter(y, method='exact')
        np.testing.assert_allclose(z, self._signal)
        # Not invertible if not a frame.
        if sys.version_info > (3, 4):
            g = filters.Expwin(self._G)
            with self.assertLogs(level='WARNING'):
                h = g.inverse()
                h.evaluate(self._G.e)

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
        f = filters.Rectangular(self._G, None, 0.1)
        f = filters.Gabor(self._G, f)
        self._test_methods(f, tight=False, check=False)
        self.assertRaises(ValueError, filters.Gabor, graphs.Sensor(), f)
        f = filters.Regular(self._G)
        self.assertRaises(ValueError, filters.Gabor, self._G, f)

    def test_modulation(self):
        f = filters.Rectangular(self._G, None, 0.1)
        # TODO: synthesis doesn't work yet.
        # f = filters.Modulation(self._G, f, modulation_first=False)
        # self._test_methods(f, tight=False, check=False)
        f = filters.Modulation(self._G, f, modulation_first=True)
        self._test_methods(f, tight=False, check=False)
        self.assertRaises(ValueError, filters.Modulation, graphs.Sensor(), f)
        f = filters.Regular(self._G)
        self.assertRaises(ValueError, filters.Modulation, self._G, f)

    def test_modulation_gabor(self):
        """Both should be equivalent for deltas centered at the eigenvalues."""
        f = filters.Rectangular(self._G, 0, 0)
        f1 = filters.Modulation(self._G, f, modulation_first=True)
        f2 = filters.Gabor(self._G, f)
        s1 = f1.filter(self._signal)
        s2 = f2.filter(self._signal)
        np.testing.assert_allclose(s1, s2, atol=1e-5)

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
        f = filters.Regular(self._G, degree=5)
        self._test_methods(f, tight=True)
        f = filters.Regular(self._G, degree=0)
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
        f = filters.Expwin(self._G, band_min=None, band_max=0.8)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=0.1, band_max=None)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=0.1, band_max=0.7)
        self._test_methods(f, tight=False)
        f = filters.Expwin(self._G, band_min=None, band_max=None)
        self._test_methods(f, tight=True)

    def test_rectangular(self):
        f = filters.Rectangular(self._G)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=None, band_max=0.8)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=0.1, band_max=None)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=0.1, band_max=0.7)
        self._test_methods(f, tight=False, check=False)
        f = filters.Rectangular(self._G, band_min=None, band_max=None)
        self._test_methods(f, tight=True, check=True)

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
