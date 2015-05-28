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
import pygsp
from pygsp import graphs

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

    def test_graphs(self):

        def test_default_graph():
            W = np.arange(16).reshape(4, 4)
            G = graphs.Graph(W, directed=False)
            self.assertEqual(G.W, sparse.lil_matrix(W)).all()
            self.assertEqual(G.A, sparse.lil_matrix(G.W > 0))
            self.assertEqual(G.N, 1)
            self.assertEqual(G.d, 0)
            self.assertEqual(G.Ne, 0)
            self.assertFalse(G.directed)
            # TODO
            # self.assertEqual(G.L, )

        def test_NNGraph():
            pass

        def test_Bunny():
            pass

        def test_Sphere():
            pass

        def test_Cube():
            pass

        def test_Grid2d():
            pass

        def test_Torus():
            pass

        def test_Comet():
            pass

        def test_LowStretchTree():
            pass

        def test_RandomRegular():
            pass

        def test_Ring():
            pass

        def test_Community():
            pass

        def test_Sensor():
            pass

        def test_Airfoil():
            pass

        def test_DavidSensorNet():
            pass

        def test_FullConnected():
            pass

        def test_Logo():
            pass

        def test_Path():
            pass

        def test_RandomRing():
            pass

        test_default_graph()
        test_NNGraph()
        test_Bunny()
        test_Sphere()
        test_Cube()
        test_Grid2d()
        test_Torus()
        test_Comet()
        test_LowStretchTree()
        test_RandomRegular()
        test_Ring()
        test_Community()
        test_Sensor()
        test_Airfoil()
        test_DavidSensorNet()
        test_FullConnected()
        test_Logo()
        test_Path()
        test_RandomRing()

    def test_dummy(self):
        """
        Dummy test.
        """
        a = np.array([1, 2])
        b = pygsp.graphs.dummy(1, a, True)
        nptest.assert_almost_equal(a, b)


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
