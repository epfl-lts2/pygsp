#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the modulename module of the pygsp package.
"""

import sys
import numpy as np
import scipy as sp
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
        W1 = np.arange(16).reshape((4, 4)) - 8
        W1 = sparse.lil_matrix(W1)
        G1 = graphs.Graph(W1)
        lap1 = np.array([[-9.,  5.5,  3.,  0.5],
                         [5.5, -4.,  0.5, -2.],
                         [3.,  0.5,  1., -4.5],
                         [0.5, -2., -4.5,  6.]])
        lap1 = sparse.lil_matrix(lap1)
        weight_check1 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': True}
        rep1 = {'lap': lap1, 'is_dir': True, 'weight_check': weight_check1}
        t1 = {'G': G1, 'rep': rep1}

        W2 = np.empty((4, 4))
        W2[0, 1] = float('NaN')
        W2[0, 2] = float('Inf')
        G2 = graphs.Graph(W2)
        weight_check2 = {'has_inf_val': True, 'has_nan_value': True,
                         'is_not_square': True, 'diag_is_not_zero': False}
        rep2 = {'lap': None, 'is_dir': True, 'weight_check': weight_check2}
        t2 = {'G': G2, 'rep': rep2}

        W3 = np.zeros((4, 4))
        G3 = graphs.Graph(W3)
        lap3 = G3.W
        weight_check3 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': False}
        rep3 = {'lap': lap3, 'is_dir': True, 'weight_check': weight_check3}
        t3 = {'G': G3, 'rep': rep3}

        W4 = np.empty((4, 4))
        np.fill_diagonal(W4, 1)
        G4 = graphs.Graph(W4)
        lap4 = sparse.lil_matrix(W4)
        weight_check4 = {'has_inf_val': False, 'has_nan_value': False,
                         'is_not_square': False, 'diag_is_not_zero': True}
        rep4 = {'lap': lap4, 'is_dir': True, 'weight_check': weight_check4}
        t4 = {'G': G4, 'rep': rep4}

        test_graphs = [t1, t2, t3, t4]

        def test_is_directed(G, rep):
            self.assertEqual(utils.is_directed(G), rep.is_dir)

        def test_estimate_lmax(G):
            operators.compute_fourier_basis(G)
            np.assert_almost_equal(utils.estimate_lmax(G)[0], G.lmax)

        def test_check_weights(G, w_c):
            self.assertEqual(utils.check_weights(G.W), w_c)

        # TODO move test_create_laplacian in Operator
        def test_create_laplacian(t):
            self.assertEqual(utils.create_laplacian(G), mat_answser)

        def test_check_connectivity(t, **kwargs):
            self.assertTrue(utils.check_connectivity(G))

        def test_distanz(x, y):
            # TODO test with matlab
            mat_answser = None
            self.assertEqual(utils.distanz(x, y))

        def test_symetrize(W, sy_type):
            # mat_answser = None
            self.assertAlmostEqual(mat_answser, utils.symetrize(W, sy_type))

        def test_tree_depths(A, root):
            # mat_answser = None
            self.assertEqual(mat_answser, utils.tree_depths(A, root))

        # Doesn't work bc of python bug
        for t in test_graphs:
            test_is_directed(t['G'], t.rep)
            test_estimate_lmax(t.G)
            test_check_weights(t.G, t.rep.check_weights)
            test_create_laplacian(t)
            test_check_connectivity(t)

        test_tree_depths(A, root)
        for s in stype:
            test_symetrize(W, s)

        test_distanz(x, y)

    def test_dummy(self):
        """
        Dummy test.
        """
        a = np.array([1, 2])
        b = utils.dummy(1, a, True)
        nptest.assert_almost_equal(a, b)


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
