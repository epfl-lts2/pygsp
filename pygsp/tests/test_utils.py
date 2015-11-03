#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the utils module of the pygsp package.
"""

import sys
import numpy as np
import numpy.testing as nptest
from scipy import sparse
from pygsp import utils, graphs, operators

# Use the unittest2 backport on Python 2.6 to profit from the new features.
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_utils(self):
        # Data init
        W1 = np.arange(16).reshape((4, 4))
        mask1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        W1[mask1 == 1] = 0
        W1 = sparse.lil_matrix(W1)
        G1 = graphs.Graph(W1)
        lap1 = np.array([[4, -1, 0, -3],
                         [-4, 10, -6, 0],
                         [0, -9, 20, -11],
                         [-12, 0, -14, 26]])
        lap1 = sparse.lil_matrix(lap1)
        sym1 = np.matrix([[0, 2.5, 0, 7.5],
                          [2.5, 0, 7.5, 0],
                          [0, 7.5, 0, 12.5],
                          [7.5, 0, 12.5, 0]])
        sym1 = sparse.lil_matrix(sym1)
        weight_check1 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': False}
        rep1 = {'lap': lap1, 'is_dir': True, 'weight_check': weight_check1,
                'is_conn': True, 'sym': sym1}
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
        lap3 = G3.W
        sym3 = G3.W
        weight_check3 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': False}
        rep3 = {'lap': lap3, 'is_dir': False, 'weight_check': weight_check3,
                'is_conn': False, 'sym': sym3}
        t3 = {'G': G3, 'rep': rep3}

        W4 = np.zeros((4, 4))
        np.fill_diagonal(W4, 1)
        G4 = graphs.Graph(W4)
        lap4 = sparse.lil_matrix(W4)
        sym4 = sparse.lil_matrix(W4)
        weight_check4 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': True}
        rep4 = {'lap': lap4, 'is_dir': False, 'weight_check': weight_check4,
                'is_conn': False, 'sym': sym4}
        t4 = {'G': G4, 'rep': rep4}

        test_graphs = [t1, t3, t4]

        def test_is_directed(G, rep):
            self.assertEqual(graphs.gutils.is_directed(G), rep['is_dir'])

        def test_estimate_lmax(G):
            operators.compute_fourier_basis(G)
            nptest.assert_almost_equal(graphs.gutils.estimate_lmax(G)[0], G.lmax)

        def test_check_weights(G, w_c):
            self.assertEqual(graphs.gutils.check_weights(G.W), w_c)

        def test_check_connectivity(G, is_conn, **kwargs):
            self.assertEqual(graphs.gutils.check_connectivity(G), is_conn)

        def test_distanz(x, y):
            # TODO test with matlab to compare
            self.assertEqual(utils.distanz(x, y))

        def test_symmetrize(W, ans):
            # mat_answser = None
            check = np.all((ans == graphs.gutils.symmetrize(W)).todense())
            self.assertTrue(check)

        # Not ready yet
        # def test_tree_depths(A, root):
        #     # mat_answser = None
        #     self.assertEqual(mat_answser, utils.tree_depths(A, root))
        for t in test_graphs:
            test_is_directed(t['G'], t['rep'])
            test_estimate_lmax(t['G'])
            test_check_weights(t['G'], t['rep']['weight_check'])
            test_check_connectivity(t['G'], t['rep']['is_conn'])
            test_symmetrize(t['G'].W, t['rep']['sym'])

        G5 = graphs.Graph(np.arange(16).reshape((4, 4)))
        checks5 = {'has_inf_val': False, 'has_nan_value': False, 'is_not_square': False, 'diag_is_not_zero': True}
        test_check_weights(G5, checks5)

        with self.assertRaises(ValueError):
            test_estimate_lmax(t2['G'])

        # Not ready yet
        # test_tree_depths(A, root)

        # test_distanz(x, y)

suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
