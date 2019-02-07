# -*- coding: utf-8 -*-

"""
Test suite for the learning module of the pygsp package.

"""

import unittest

import numpy as np

from pygsp import graphs, filters, learning


class TestCase(unittest.TestCase):

    def test_regression_tikhonov_1(self):
        """Solve a trivial regression problem."""
        G = graphs.Ring(N=8)
        signal = np.array([0, np.nan, 4, np.nan, 4, np.nan, np.nan, np.nan])
        mask = np.array([True, False, True, False, True, False, False, False])
        truth = np.array([0, 2, 4, 4, 4, 3, 2, 1])
        recovery = learning.regression_tikhonov(G, signal, mask, tau=0)
        np.testing.assert_allclose(recovery, truth)

        # Test the numpy solution.
        G = graphs.Graph(G.W.toarray())
        recovery = learning.regression_tikhonov(G, signal, mask, tau=0)
        np.testing.assert_allclose(recovery, truth)

    def test_regression_tikhonov_2(self):
        """Solve a regression problem with a constraint."""
        G = graphs.Sensor(100)
        G.estimate_lmax()

        # Create a smooth signal.
        filt = filters.Filter(G, lambda x: 1 / (1 + 10*x))
        rs = np.random.RandomState(1)
        signal = filt.analyze(rs.normal(size=(G.n_vertices, 5)))

        # Make the input signal.
        mask = rs.uniform(0, 1, [G.n_vertices]) > 0.5
        measures = signal.copy()
        measures[~mask] = np.nan

        # Solve the problem.
        recovery0 = learning.regression_tikhonov(G, measures, mask, tau=0)
        recovery1 = np.zeros_like(recovery0)
        for i in range(recovery0.shape[1]):
            recovery1[:, i] = learning.regression_tikhonov(
                G, measures[:, i], mask, tau=0)

        G = graphs.Graph(G.W.toarray())
        recovery2 = learning.regression_tikhonov(G, measures, mask, tau=0)
        recovery3 = np.zeros_like(recovery0)
        for i in range(recovery0.shape[1]):
            recovery3[:, i] = learning.regression_tikhonov(
                G, measures[:, i], mask, tau=0)

        np.testing.assert_allclose(recovery1, recovery0)
        np.testing.assert_allclose(recovery2, recovery0)
        np.testing.assert_allclose(recovery3, recovery0)

    def test_regression_tikhonov_3(self, tau=3.5):
        """Solve a relaxed regression problem."""
        G = graphs.Sensor(100)
        G.estimate_lmax()

        # Create a smooth signal.
        filt = filters.Filter(G, lambda x: 1 / (1 + 10*x))
        rs = np.random.RandomState(1)
        signal = filt.analyze(rs.normal(size=(G.n_vertices, 6)))

        # Make the input signal.
        mask = rs.uniform(0, 1, G.n_vertices) > 0.5
        measures = signal.copy()
        measures[~mask] = 0

        L = G.L.toarray()
        recovery = np.matmul(np.linalg.inv(np.diag(1*mask) + tau * L),
                             (mask * measures.T).T)

        # Solve the problem.
        recovery0 = learning.regression_tikhonov(G, measures, mask, tau=tau)
        recovery1 = np.zeros_like(recovery0)
        for i in range(recovery0.shape[1]):
            recovery1[:, i] = learning.regression_tikhonov(
                G, measures[:, i], mask, tau)

        G = graphs.Graph(G.W.toarray())
        recovery2 = learning.regression_tikhonov(G, measures, mask, tau)
        recovery3 = np.zeros_like(recovery0)
        for i in range(recovery0.shape[1]):
            recovery3[:, i] = learning.regression_tikhonov(
                G, measures[:, i], mask, tau)

        np.testing.assert_allclose(recovery0, recovery, atol=1e-5)
        np.testing.assert_allclose(recovery1, recovery, atol=1e-5)
        np.testing.assert_allclose(recovery2, recovery, atol=1e-5)
        np.testing.assert_allclose(recovery3, recovery, atol=1e-5)

    def test_classification_tikhonov(self):
        """Solve a classification problem."""
        G = graphs.Logo()
        signal = np.zeros([G.n_vertices], dtype=int)
        signal[G.info['idx_s']] = 1
        signal[G.info['idx_p']] = 2

        # Make the input signal.
        rs = np.random.RandomState(seed=1)
        mask = rs.uniform(size=G.n_vertices) > 0.3

        measures = signal.copy()
        measures[~mask] = -1

        # Solve the classification problem.
        recovery = learning.classification_tikhonov(G, measures, mask, tau=0)
        recovery = np.argmax(recovery, axis=1)

        np.testing.assert_array_equal(recovery, signal)

        # Test the function with the simplex projection.
        recovery = learning.classification_tikhonov_simplex(
            G, measures, mask, tau=0.1)

        # Assert that the probabilities sums to 1
        np.testing.assert_allclose(np.sum(recovery, axis=1), 1)

        # Check the quality of the solution.
        recovery = np.argmax(recovery, axis=1)
        np.testing.assert_allclose(signal, recovery)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
