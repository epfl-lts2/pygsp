#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for the plotting module of the pygsp package."""

import sys
import numpy as np
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

    def test_plotting(self):

        def needed_attributes_testing(G):
            self.assertTrue(hasattr(G, 'coords'))
            self.assertTrue(hasattr(G, 'A'))
            self.assertEqual(G.N, G.coords.shape[0])

        def test_default_graph():
            W = np.arange(16).reshape(4, 4)
            G = graphs.Graph(W)
            ki, kj = np.nonzero(G.A)
            self.assertEqual(ki.shape[0], G.Ne)
            self.assertEqual(kj.shape[0], G.Ne)
            needed_attributes_testing(G)

        def test_NNGraph():
            Xin = np.arange(90).reshape(30, 3)
            G = graphs.NNGraph(Xin)
            needed_attributes_testing(G)

        def test_Bunny():
            G = graphs.Bunny()
            needed_attributes_testing(G)

        def test_Cube():
            G = graphs.Cube()
            G2 = graphs.Cube(nb_dim=2)
            needed_attributes_testing(G)

            needed_attributes_testing(G2)

        def test_Sphere():
            G = graphs.Sphere()
            needed_attributes_testing(G)

        def test_TwoMoons():
            G = graphs.TwoMoons()
            G2 = graphs.TwoMoons(moontype='synthetised')
            needed_attributes_testing(G)

            needed_attributes_testing(G2)

        def test_Grid2d():
            G = graphs.Grid2d()
            needed_attributes_testing(G)

        def test_Torus():
            G = graphs.Torus()
            needed_attributes_testing(G)

        def test_Comet():
            G = graphs.Comet()
            needed_attributes_testing(G)

        def test_LowStretchTree():
            G = graphs.LowStretchTree()
            needed_attributes_testing(G)

        def test_RandomRegular():
            G = graphs.RandomRegular()
            needed_attributes_testing(G)

        def test_Ring():
            G = graphs.Ring()
            needed_attributes_testing(G)

        def test_Community():
            G = graphs.Community()
            needed_attributes_testing(G)

        def test_Minnesota():
            G = graphs.Minnesota()
            needed_attributes_testing(G)

        def test_Sensor():
            G = graphs.Sensor()
            needed_attributes_testing(G)

        def test_Airfoil():
            G = graphs.Airfoil()
            needed_attributes_testing(G)

        def test_DavidSensorNet():
            G = graphs.DavidSensorNet()
            G2 = graphs.DavidSensorNet(N=500)
            G3 = graphs.DavidSensorNet(N=128)

            needed_attributes_testing(G)
            needed_attributes_testing(G2)
            needed_attributes_testing(G3)

        def test_FullConnected():
            G = graphs.FullConnected()
            needed_attributes_testing(G)

        def test_Logo():
            G = graphs.Logo()
            needed_attributes_testing(G)

        def test_Path():
            G = graphs.Path()
            needed_attributes_testing(G)

        def test_RandomRing():
            G = graphs.RandomRing()
            needed_attributes_testing(G)

        def test_SwissRoll():
            G = graphs.SwissRoll()
            needed_attributes_testing(G)

suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
