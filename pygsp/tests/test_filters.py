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

    def _test_methods(self, f, tight, check=True):
        self.assertIs(f.G, self._G)

        f.evaluate(self._G.e)
        f.evaluate(np.random.normal(size=(4, 6, 3)))

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


class TestApproximations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = graphs.Logo()
        cls._G.compute_fourier_basis()
        cls._rs = np.random.RandomState(42)
        cls._signal = cls._rs.uniform(size=cls._G.N)

    def test_scaling(self, N=40):
        x = np.linspace(0, self._G.lmax, N)
        x = filters.Chebyshev.scale_data(x, self._G.lmax)
        self.assertEqual(x.min(), -1)
        self.assertEqual(x.max(), 1)
        L = filters.Chebyshev.scale_operator(self._G.L, self._G.lmax)

    def test_chebyshev_basis(self, K=5, c=2, N=100):
        r"""
        Test that the evaluation of the Chebyshev series yields the expected
        basis. We only test the first two elements here. The higher-order
        polynomials are compared with the trigonometric definition.
        """
        f = filters.Chebyshev(self._G, c * np.identity(K))
        x = np.linspace(0, self._G.lmax, N)
        y = f.evaluate(x)
        np.testing.assert_equal(y[0], c)
        np.testing.assert_allclose(y[1], np.linspace(-c, c, N))

    def test_evaluation_methods(self, K=30, F=5, N=100):
        r"""Test that all evaluation methods return the same results."""
        coefficients = self._rs.uniform(size=(K, F, F))
        f = filters.Chebyshev(self._G, coefficients)
        x = np.linspace(0, self._G.lmax, N)
        y1 = f.evaluate(x, method='recursive')
        y2 = f.evaluate(x, method='direct')
        y3 = f.evaluate(x, method='clenshaw')
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(y1, y3)
        # Evaluate on n-dimensional arrays.
        x = self._rs.uniform(0, self._G.lmax, size=(3, 1, 19))
        y1 = f.evaluate(x, method='recursive')
        y2 = f.evaluate(x, method='direct')
        y3 = f.evaluate(x, method='clenshaw')
        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(y1, y3)
        # Unknown method.
        self.assertRaises(ValueError, f.evaluate, x, method='unk')

    def test_filter_identity(self, M=10, c=2.3):
        r"""Test that filtering with c0 only scales the signal."""
        x = self._rs.uniform(size=(M, self._G.N))
        f = filters.Chebyshev(self._G, c)
        y = f.filter(x, method='recursive')
        np.testing.assert_equal(y, c * x)
        # Test with dense Laplacian.
        L = self._G.L
        self._G.L = L.toarray()
        y = f.filter(x, method='recursive')
        self._G.L = L
        np.testing.assert_equal(y, c * x)

    def test_filter_methods(self, K=30, Fin=5, Fout=6, M=100):
        r"""Test that all filter methods return the same results."""
        coefficients = self._rs.uniform(size=(K, Fout, Fin))
        x = self._rs.uniform(size=(M, Fin, self._G.N))
        f = filters.Chebyshev(self._G, coefficients)
        y1 = f.filter(x, method='recursive')
        y2 = f.filter(x, method='clenshaw')
        self.assertTupleEqual(y1.shape, (M, Fout, self._G.N))
        np.testing.assert_allclose(y1, y2, rtol=1e-5)
        # Unknown method.
        self.assertRaises(ValueError, f.filter, x, method='unk')

    def test_coefficients(self, K=10, slope=3.14):
        r"""Test that the computed Chebyshev coefficients are correct."""
        # Identity filter.
        f = filters.Heat(self._G, tau=0)
        f = filters.Chebyshev.from_filter(f, order=K)
        c = f._coefficients.squeeze()
        np.testing.assert_allclose(c, [1] + K*[0], atol=1e-12)
        # Linear filter.
        f = filters.Filter(self._G, lambda x: slope*x)
        f = filters.Chebyshev.from_filter(f, order=K)
        c1 = f._coefficients.squeeze()
        c2 = slope / 2 * self._G.lmax
        np.testing.assert_allclose(c1, 2*[c2] + (K-1)*[0], atol=1e-12)

    def test_approximations(self, N=100, K=20):
        r"""Test that the approximations are not far from the exact filters."""
        # Evaluation.
        x = self._rs.uniform(0, self._G.lmax, N)
        f1 = filters.Heat(self._G)
        y1 = f1.evaluate(x)
        f2 = f1.approximate('Chebyshev', order=K)
        y2 = f2.evaluate(x)
        np.testing.assert_allclose(y2, y1.squeeze())
        # Filtering.
        x = self._rs.uniform(size=(1, 1, self._G.N))
        y1 = f1.filter(x.T).T
        y2 = f2.filter(x)
        np.testing.assert_allclose(y2.squeeze(), y1)

    def test_shape_normalization(self):
        """Test that signal's shapes are properly normalized."""
        # TODO: should also test filters which are not approximations.

        def test_normalization(M, Fin, Fout, K=7):

            def test_shape(y, M, Fout, N=self._G.N):
                """Test that filtered signals are squeezed."""
                if Fout == 1 and M == 1:
                    self.assertEqual(y.shape, (N,))
                elif Fout == 1:
                    self.assertEqual(y.shape, (M, N))
                elif M == 1:
                    self.assertEqual(y.shape, (Fout, N))
                else:
                    self.assertEqual(y.shape, (M, Fout, N))

            coefficients = self._rs.uniform(size=(K, Fout, Fin))
            f = filters.Chebyshev(self._G, coefficients)
            assert f.shape == (Fin, Fout)
            assert (f.n_features_in, f.n_features_out) == (Fin, Fout)

            x = self._rs.uniform(size=(M, Fin, self._G.N))
            y = f.filter(x)
            test_shape(y, M, Fout)

            if Fin == 1 or M == 1:
                # It only makes sense to squeeze if one dimension is unitary.
                x = x.squeeze()
                y = f.filter(x)
                test_shape(y, M, Fout)

        # Test all possible correct combinations of input and output signals.
        for M in [1, 9]:
            for Fin in [1, 3]:
                for Fout in [1, 5]:
                    test_normalization(M, Fin, Fout)

        # Test failure cases.
        M, Fin, Fout, K = 9, 3, 5, 7
        coefficients = self._rs.uniform(size=(K, Fout, Fin))
        f = filters.Chebyshev(self._G, coefficients)
        x = self._rs.uniform(size=(M, Fin, 2))
        self.assertRaises(ValueError, f.filter, x)
        x = self._rs.uniform(size=(M, 2, self._G.N))
        self.assertRaises(ValueError, f.filter, x)
        x = self._rs.uniform(size=(2, self._G.N))
        self.assertRaises(ValueError, f.filter, x)
        x = self._rs.uniform(size=(self._G.N))
        self.assertRaises(ValueError, f.filter, x)
        x = self._rs.uniform(size=(2, M, Fin, self._G.N))
        self.assertRaises(ValueError, f.filter, x)

suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
suite_approximations = unittest.TestLoader().loadTestsFromTestCase(TestApproximations)
