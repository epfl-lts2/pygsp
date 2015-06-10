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
        W1 = sparse.lil_matrix(W)
        G1 = graphs.Graph(W)
        t1 = {'G': G1, 'lap': None, 'is_dir': True, 'diag_is_not_zero': True}

        W2 = np.empty((4, 5))
        W2[0,1] = float('NaN')
        W2[0,2] = float('Inf')
        G2 = graphs.Graph(W2)
        t2 = {'G': G2, 'lap': None, 'is_dir': True, 'has_nan_val': True, 'has_inf_val': True}

        W3 = np.zeros((4, 4))
        G3 = graphs.Graph(W3)
        t3 = {'G': G3, 'lap': None, 'is_dir': True}

        W4 = np.empty((4, 4))
        np.fill_diagonal(W4, 1)
        G4 = graphs.Graph(W4)
        t4 = {'G': G4, 'lap': None, 'is_dir': True, 'diag_is_not_zero': True}

        graphs_test = [t1, t2, t3, t4]
        for t in graphs_test:
            t['lmax'] = np.max(t.G.lmax)

        # TODO choose values
        x = None
        y = None
        stype = ['average', 'full']

        def test_is_directed(t):
            self.assertEqual(utils.is_directed(G), t.is_dir)

        def test_estimate_lmax(G):
            operators.compute_fourier_basis(G)
            np.assert_almost_equal(utils.estimate_lmax(G)[0], G.lmax)

        def test_check_weights(t):
            # self.assertAlmostEqual(utils.check_weights(t.G.W), t)
            pass

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
        for t in graphs_test:
            test_is_directed(t)
            test_estimate_lmax(t.G)
            test_check_weights(t)
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
