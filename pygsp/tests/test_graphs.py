# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

import unittest

import numpy as np
from skimage import data, img_as_float

from pygsp import graphs


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._img = img_as_float(data.camera()[::16, ::16])

    @classmethod
    def tearDownClass(cls):
        pass

    def test_graph(self):
        W = np.arange(16).reshape(4, 4)
        G = graphs.Graph(W)
        np.testing.assert_allclose(G.W.A, W)
        np.testing.assert_allclose(G.A.A, G.W.A > 0)
        self.assertEqual(G.N, 4)
        np.testing.assert_allclose(G.d, np.array([3, 4, 4, 4]))
        self.assertEqual(G.Ne, 15)
        self.assertTrue(G.is_directed())
        ki, kj = np.nonzero(G.A)
        self.assertEqual(ki.shape[0], G.Ne)
        self.assertEqual(kj.shape[0], G.Ne)

    def test_laplacian(self):
        # TODO: should test correctness.

        G = graphs.StochasticBlockModel(undirected=True)
        self.assertFalse(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        G.compute_laplacian(lap_type='normalized')

        G = graphs.StochasticBlockModel(undirected=False)
        self.assertTrue(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        self.assertRaises(NotImplementedError, G.compute_laplacian,
                          lap_type='normalized')

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
        graphs.Ring(N=32, k=16)

    def test_community(self):
        graphs.Community()
        graphs.Community(comm_density=0.2)
        graphs.Community(k_neigh=5)
        graphs.Community(world_density=0.8)

    def test_minnesota(self):
        graphs.Minnesota()

    def test_sensor(self):
        graphs.Sensor(regular=True)
        graphs.Sensor(regular=False)
        graphs.Sensor(distribute=True)
        graphs.Sensor(distribute=False)
        graphs.Sensor(connected=True)
        graphs.Sensor(connected=False)

    def test_stochasticblockmodel(self):
        graphs.StochasticBlockModel(undirected=True)
        graphs.StochasticBlockModel(undirected=False)
        graphs.StochasticBlockModel(no_self_loop=True)
        graphs.StochasticBlockModel(no_self_loop=False)

    def test_airfoil(self):
        graphs.Airfoil()

    def test_davidsensornet(self):
        graphs.DavidSensorNet()
        graphs.DavidSensorNet(N=500)
        graphs.DavidSensorNet(N=128)

    def test_erdosreny(self):
        graphs.ErdosRenyi(connected=False)
        graphs.ErdosRenyi(connected=True)
        graphs.ErdosRenyi(directed=False)
        # graphs.ErdosRenyi(directed=True)  # TODO: bug in implementation

    def test_fullconnected(self):
        graphs.FullConnected()

    def test_logo(self):
        graphs.Logo()

    def test_path(self):
        graphs.Path()

    def test_randomring(self):
        graphs.RandomRing()

    def test_swissroll(self):
        graphs.SwissRoll(srtype='uniform')
        graphs.SwissRoll(srtype='classic')
        graphs.SwissRoll(noise=True)
        graphs.SwissRoll(noise=False)
        graphs.SwissRoll(dim=2)
        graphs.SwissRoll(dim=3)

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


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
