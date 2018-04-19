# -*- coding: utf-8 -*-

"""
Test suite for the utils module of the pygsp package.

"""

import unittest

import numpy as np
from scipy import sparse

from pygsp import graphs, utils


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_symmetrize(self):
        W = sparse.random(100, 100, random_state=42)
        for method in ['average', 'maximum', 'fill', 'tril', 'triu']:
            # Test that the regular and sparse versions give the same result.
            W1 = utils.symmetrize(W, method=method)
            W2 = utils.symmetrize(W.toarray(), method=method)
            np.testing.assert_equal(W1.toarray(), W2)
        self.assertRaises(ValueError, utils.symmetrize, W, 'sum')

    def test_utils(self):
        # Data init
        W1 = np.arange(16).reshape((4, 4))
        mask1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                          [1, 0, 1, 0], [0, 1, 0, 1]])
        W1[mask1 == 1] = 0
        W1 = sparse.lil_matrix(W1)
        G1 = graphs.Graph(W1)
        lap1 = np.array([[10, -2.5, 0, -7.5],
                         [-2.5, 10, -7.5, 0],
                         [0, -7.5, 20, -12.5],
                         [-7.5, 0, -12.5, 20]])

        sym1 = np.matrix([[0, 2.5, 0, 7.5],
                          [2.5, 0, 7.5, 0],
                          [0, 7.5, 0, 12.5],
                          [7.5, 0, 12.5, 0]])
        sym1 = sparse.lil_matrix(sym1)
        weight_check1 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': False}
        rep1 = {'lap': lap1, 'is_dir': True, 'weight_check': weight_check1,
                'is_conn': True, 'sym': sym1, 'lmax': 35.}
        t1 = {'G': G1, 'rep': rep1}

        W2 = np.zeros((4, 4))
        W2[0, 1] = float('NaN')
        W2[0, 2] = float('Inf')
        G2 = graphs.Graph(W2)
        weight_check2 = {'has_inf_val': True, 'has_nan_value': True,
                         'is_not_square': True, 'diag_is_not_zero': False}
        rep2 = {'lap': None, 'is_dir': True, 'weight_check': weight_check2,
                'is_conn': False}
        t2 = {'G': G2, 'rep': rep2}

        W3 = np.zeros((4, 4))
        G3 = graphs.Graph(W3)
        lap3 = W3
        sym3 = G3.W
        weight_check3 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': False}
        rep3 = {'lap': lap3, 'is_dir': False, 'weight_check': weight_check3,
                'is_conn': False, 'sym': sym3, 'lmax': 0.}
        t3 = {'G': G3, 'rep': rep3}

        W4 = np.zeros((4, 4))
        np.fill_diagonal(W4, 1)
        G4 = graphs.Graph(W4)
        lap4 = np.zeros((4, 4))
        sym4 = sparse.csc_matrix(W4)
        weight_check4 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': True}
        rep4 = {'lap': lap4, 'is_dir': False, 'weight_check': weight_check4,
                'is_conn': False, 'sym': sym4, 'lmax': 0.}
        t4 = {'G': G4, 'rep': rep4}

        test_graphs = [t1, t3, t4]

        def test_is_directed(G, is_dir):
            self.assertEqual(G.is_directed(), is_dir)

        def test_laplacian(G, lap):
            self.assertTrue((G.L == lap).all())

        def test_estimate_lmax(G, lmax):
            G.estimate_lmax()
            self.assertTrue(lmax <= G.lmax and G.lmax <= 1.02 * lmax)

        def test_check_weights(G, w_c):
            self.assertEqual(G.check_weights(), w_c)

        def test_is_connected(G, is_conn, **kwargs):
            self.assertEqual(G.is_connected(), is_conn)

        def test_distanz(x, y):
            # TODO test with matlab to compare
            self.assertEqual(utils.distanz(x, y))

        # Not ready yet
        # def test_tree_depths(A, root):
        #     # mat_answser = None
        #     self.assertEqual(mat_answser, utils.tree_depths(A, root))
        for t in test_graphs:
            test_is_directed(t['G'], t['rep']['is_dir'])
            test_laplacian(t['G'], t['rep']['lap'])
            test_estimate_lmax(t['G'], t['rep']['lmax'])
            test_check_weights(t['G'], t['rep']['weight_check'])
            test_is_connected(t['G'], t['rep']['is_conn'])

        G5 = graphs.Graph(np.arange(16).reshape((4, 4)))
        checks5 = {'has_inf_val': False, 'has_nan_value': False,
                   'is_not_square': False, 'diag_is_not_zero': True}
        test_check_weights(G5, checks5)

        # Not ready yet
        # test_tree_depths(A, root)

        # test_distanz(x, y)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
