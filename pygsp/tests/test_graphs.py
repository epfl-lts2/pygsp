#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

import unittest

import numpy as np
from skimage import data, img_as_float

from pygsp import graphs


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self._img = img_as_float(data.camera()[::16, ::16])

    def tearDown(self):
        pass

    def test_default_graph(self):
        W = np.arange(16).reshape(4, 4)
        G = graphs.Graph(W)
        assert np.allclose(G.W.A, W)
        assert np.allclose(G.A.A, G.W.A > 0)
        self.assertEqual(G.N, 4)
        assert np.allclose(G.d, np.array([[3], [4], [4], [4]]))
        self.assertEqual(G.Ne, 15)
        self.assertTrue(G.directed)
        ki, kj = np.nonzero(G.A)
        self.assertEqual(ki.shape[0], G.Ne)
        self.assertEqual(kj.shape[0], G.Ne)

    def test_nngraph(self):
        Xin = np.arange(90).reshape(30, 3)
        dist_types = ['euclidean', 'manhattan', 'max_dist', 'minkowski']

        for dist_type in dist_types:

            # Only p-norms with 1<=p<=infinity permitted.
            if dist_type != 'minkowski':
                graphs.NNGraph(Xin, NNtype='radius', dist_type=dist_type)
                graphs.NNGraph(Xin, NNtype='knn', dist_type=dist_type)

            # Distance type unsupported in the C bindings,
            # use the C++ bindings instead.
            if dist_type != 'max_dist':
                graphs.NNGraph(Xin, use_flann=True, NNtype='knn',
                               dist_type=dist_type)

    def test_bunny(self):
        graphs.Bunny()

    def test_cube(self):
        graphs.Cube()
        graphs.Cube(nb_dim=2)

    def test_sphere(self):
        graphs.Sphere()

    def test_twomoons(self):
        graphs.TwoMoons()
        graphs.TwoMoons(moontype='synthesized')

    def test_torus(self):
        graphs.Torus()

    def test_comet(self):
        graphs.Comet()

    def test_lowstretchtree(self):
        graphs.LowStretchTree()

    def test_randomregular(self):
        graphs.RandomRegular()

    def test_ring(self):
        graphs.Ring()

    def test_community(self):
        graphs.Community()

    def test_minnesota(self):
        graphs.Minnesota()

    def test_sensor(self):
        graphs.Sensor()

    def test_airfoil(self):
        graphs.Airfoil()

    def test_davidsensornet(self):
        graphs.DavidSensorNet()
        graphs.DavidSensorNet(N=500)
        graphs.DavidSensorNet(N=128)

    def test_fullconnected(self):
        graphs.FullConnected()

    def test_logo(self):
        graphs.Logo()

    def test_path(self):
        graphs.Path()

    def test_randomring(self):
        graphs.RandomRing()

    def test_swissroll(self):
        graphs.SwissRoll()

    def test_grid2d(self):
        G = graphs.Grid2d(shape=(3, 2))
        self.assertEqual([G.h, G.w], [3, 2])
        G = graphs.Grid2d(shape=(3,))
        self.assertEqual([G.h, G.w], [3, 3])
        G = graphs.Grid2d(shape=3)
        self.assertEqual([G.h, G.w], [3, 3])

    def test_imgpatches(self):
        graphs.ImgPatches(img=self._img, patch_shape=(3, 3))

    def test_grid2dimgpatches(self):
        graphs.Grid2dImgPatches(img=self._img, patch_shape=(3, 3))


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)
