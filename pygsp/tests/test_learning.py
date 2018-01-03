# -*- coding: utf-8 -*-

"""
Test suite for the filters module of the pygsp package.

"""

import unittest

import numpy as np

from pygsp import graphs, filters, learning


class TestCase(unittest.TestCase):

    def test_regression_tik(self):
        # Create the graph
        G = graphs.Sensor(N=100)
        G.estimate_lmax()

        # Create a smooth signal
        def filt(x):
            return 1 / (1+10*x)
        g_filt = filters.Filter(G, filt)
        sig = g_filt.analyze(np.random.randn(G.N, 5))

        # Make the input signal
        M = np.random.uniform(0, 1, [G.N]) > 0.5  # Mask

        measurements = sig.copy()
        measurements[M == False] = np.nan

        # Solve the problem

        recovery0 = learning.regression_tik(G, measurements, M, tau=0)
        recovery1 = np.zeros(recovery0.shape)
        for i in range(recovery0.shape[1]):
            recovery1[:, i] = learning.regression_tik(G, measurements[:, i], M, tau=0)

        G2 = graphs.Graph(G.W.toarray())
        G2.estimate_lmax()
        recovery2 = learning.regression_tik(G2, measurements, M, tau=0)
        recovery3 = np.zeros(recovery0.shape)
        for i in range(recovery0.shape[1]):
            recovery3[:, i] = learning.regression_tik(G2, measurements[:, i], M, tau=0)
        # print(np.linalg.norm(recovery0-recovery1,ord='fro'))
        np.testing.assert_allclose(recovery0, recovery1, rtol=1e-10, atol=1e-10)
        # print(np.linalg.norm(recovery0-recovery2,ord='fro'))
        np.testing.assert_allclose(recovery0, recovery2, rtol=1e-10, atol=1e-10)
        # print(np.linalg.norm(recovery0-recovery3,ord='fro'))
        np.testing.assert_allclose(recovery0, recovery3, rtol=1e-10, atol=1e-10)

        # Solve the problem for a trivial case
        G3 = graphs.Ring(N=8)
        sig = np.array([0, np.nan, 4, np.nan, 4, np.nan, np.nan, np.nan])
        M = np.array([True, False, True, False, True, False, False, False])
        sig_rec = np.array([0, 2, 4, 4, 4, 3, 2, 1])
        sig_rec2 = learning.regression_tik(G3, sig, M, tau=0)
        # print(np.linalg.norm(sig_rec-sig_rec2))
        np.testing.assert_allclose(sig_rec, sig_rec2)

    def test_regression_tik2(self):
        tau = 3.5

        # Create the graph
        G = graphs.Sensor(N=100)
        G.estimate_lmax()

        # Create a smooth signal
        def filt(x):
            return 1 / (1+10*x)
        g_filt = filters.Filter(G, filt)
        sig = g_filt.analyze(np.random.randn(G.N, 6))

        # Make the input signal
        M = np.random.uniform(0, 1, [G.N]) > 0.5  # Mask

        measurements = sig.copy()
        measurements[M == False] = 0

        L = G.L.toarray()
        recovery = np.matmul(np.linalg.inv(np.diag(1*M) + tau * L), (M * measurements.T).T)

        # Solve the problem
        recovery0 = learning.regression_tik(G, measurements, M, tau=tau)
        recovery1 = np.zeros(recovery0.shape)
        for i in range(recovery0.shape[1]):
            recovery1[:, i] = learning.regression_tik(G, measurements[:, i], M, tau=tau)

        G2 = graphs.Graph(G.W.toarray())
        G2.estimate_lmax()
        recovery2 = learning.regression_tik(G2, measurements, M, tau=tau)
        recovery3 = np.zeros(recovery0.shape)
        for i in range(recovery0.shape[1]):
            recovery3[:, i] = learning.regression_tik(
                G2, measurements[:, i], M, tau=tau)
        # print(np.linalg.norm(recovery-recovery0,ord='fro'))
        # print(np.linalg.norm(recovery-recovery1,ord='fro'))
        # print(np.linalg.norm(recovery-recovery2,ord='fro'))
        # print(np.linalg.norm(recovery-recovery3,ord='fro'))
        np.testing.assert_allclose(recovery, recovery0,rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(recovery, recovery1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(recovery, recovery2, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(recovery, recovery3, rtol=1e-5, atol=1e-6)

    def test_classification_tik(self):
        G = graphs.Logo()
        idx_g = np.squeeze(G.info['idx_g'])
        idx_s = np.squeeze(G.info['idx_s'])
        idx_p = np.squeeze(G.info['idx_p'])
        sig = np.zeros([G.N], dtype=int)
        sig[idx_s] = 1
        sig[idx_p] = 2

        # Make the input signal
        np.random.seed(seed=1)
        M = np.random.uniform(0,1,[G.N])>0.3 

        measurements = sig.copy()
        measurements[M==False] = -1

        # Solve the classification problem
        recovery = learning.classification_tik(G, measurements, M, tau=0)

        # print(sum(np.abs(recovery-sig)))
        np.testing.assert_array_equal(recovery, sig)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
