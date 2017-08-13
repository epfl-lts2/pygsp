#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for the graphs module of the pygsp package."""

import unittest
import numpy as np
from scipy import sparse
from pygsp import graphs


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
            dist_types = ['euclidean', 'manhattan', 'max_dist', 'minkowski']
            for dist_type in dist_types:
                G1 = graphs.NNGraph(Xin, NNtype='knn', dist_type=dist_type)
                G2 = graphs.NNGraph(Xin, use_flann=True, NNtype='knn',
                                    dist_type=dist_type)
                G3 = graphs.NNGraph(Xin, NNtype='radius', dist_type=dist_type)

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

        def test_Grid2d():
            G = graphs.Grid2d(shape=(3, 2))
            self.assertEqual([G.h, G.w], [3, 2])
            G = graphs.Grid2d(shape=(3,))
            self.assertEqual([G.h, G.w], [3, 3])
            G = graphs.Grid2d(shape=3)
            self.assertEqual([G.h, G.w], [3, 3])

        def test_ImgPatches():
            from skimage import data, img_as_float
            img = img_as_float(data.camera()[::16, ::16])
            G = graphs.ImgPatches(img=img, patch_shape=(3, 3))

        def test_Grid2dImgPatches():
            from skimage import data, img_as_float
            img = img_as_float(data.camera()[::16, ::16])
            G = graphs.Grid2dImgPatches(img=img, patch_shape=(3, 3))


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
