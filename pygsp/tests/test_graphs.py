#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for the graphs module of the pygsp package."""

import sys
import numpy as np
from scipy import sparse
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
            G = graphs.Graph(W)
            self.assertEqual(G.W, sparse.lil_matrix(W))
            self.assertEqual(G.A, G.W > 0)
            self.assertEqual(G.N, 4)
            self.assertEqual(G.d, [3, 4, 4, 4])
            self.assertEqual(G.Ne, 15)
            self.assertTrue(G.directed)

        def test_NNGraph():
            Xin = np.arange(90).reshape(30, 3)
            G = graphs.NNGraph(Xin)

        def test_Bunny():
            G = graphs.Bunny()

        def test_Cube():
            G = graphs.Cube()
            G2 = graphs.Cube(nb_dim=2)

        def test_Sphere():
            G = graphs.Sphere()

        def test_TwoMoons():
            G = graphs.TwoMoons()
            G2 = graphs.TwoMoons(moontype='synthetised')

        def test_Grid2d():
            G = graphs.Grid2d()

        def test_Torus():
            G = graphs.Torus()

        def test_Comet():
            G = graphs.Comet()

        def test_LowStretchTree():
            G = graphs.LowStretchTree()

        def test_RandomRegular():
            G = graphs.RandomRegular()

        def test_Ring():
            G = graphs.Ring()

        def test_Community():
            G = graphs.Community()

        def test_Minnesota():
            G = graphs.Minnesota()

        def test_Sensor():
            G = graphs.Sensor()

        def test_Airfoil():
            G = graphs.Airfoil()

        def test_DavidSensorNet():
            G = graphs.DavidSensorNet()
            G2 = graphs.DavidSensorNet(N=500)
            G3 = graphs.DavidSensorNet(N=128)

        def test_FullConnected():
            G = graphs.FullConnected()

        def test_Logo():
            G = graphs.Logo()

        def test_Path():
            G = graphs.Path()

        def test_RandomRing():
            G = graphs.RandomRing()

        def test_SwissRoll():
            G = graphs.SwissRoll()

suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
