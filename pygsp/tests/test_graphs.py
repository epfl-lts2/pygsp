# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

import unittest

import numpy as np
import scipy.linalg
from skimage import data, img_as_float

from pygsp import graphs


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = graphs.Logo()
        cls._G.compute_fourier_basis()

        cls._rs = np.random.RandomState(42)
        cls._signal = cls._rs.uniform(size=cls._G.N)

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

    def test_fourier_basis(self):
        # Smallest eigenvalue close to zero.
        np.testing.assert_allclose(self._G.e[0], 0, atol=1e-12)
        # First eigenvector is constant.
        N = self._G.N
        np.testing.assert_allclose(self._G.U[:, 0], np.sqrt(N) / N)
        # Control eigenvector direction.
        # assert (self._G.U[0, :] > 0).all()
        # Spectrum bounded by [0, 2] for the normalized Laplacian.
        G = graphs.Logo(lap_type='normalized')
        G.compute_fourier_basis()
        assert G.e[-1] < 2

    def test_eigendecompositions(self):
        G = graphs.Logo()
        U1, e1, V1 = scipy.linalg.svd(G.L.toarray())
        U2, e2, V2 = np.linalg.svd(G.L.toarray())
        e3, U3 = np.linalg.eig(G.L.toarray())
        e4, U4 = scipy.linalg.eig(G.L.toarray())
        e5, U5 = np.linalg.eigh(G.L.toarray())
        e6, U6 = scipy.linalg.eigh(G.L.toarray())

        def correct_sign(U):
            signs = np.sign(U[0, :])
            signs[signs == 0] = 1
            return U * signs
        U1 = correct_sign(U1)
        U2 = correct_sign(U2)
        U3 = correct_sign(U3)
        U4 = correct_sign(U4)
        U5 = correct_sign(U5)
        U6 = correct_sign(U6)
        V1 = correct_sign(V1.T)
        V2 = correct_sign(V2.T)

        inds3 = np.argsort(e3)[::-1]
        inds4 = np.argsort(e4)[::-1]
        np.testing.assert_allclose(e2, e1)
        np.testing.assert_allclose(e3[inds3], e1, atol=1e-12)
        np.testing.assert_allclose(e4[inds4], e1, atol=1e-12)
        np.testing.assert_allclose(e5[::-1], e1, atol=1e-12)
        np.testing.assert_allclose(e6[::-1], e1, atol=1e-12)
        np.testing.assert_allclose(U2, U1, atol=1e-12)
        np.testing.assert_allclose(V1, U1, atol=1e-12)
        np.testing.assert_allclose(V2, U1, atol=1e-12)
        np.testing.assert_allclose(U3[:, inds3], U1, atol=1e-10)
        np.testing.assert_allclose(U4[:, inds4], U1, atol=1e-10)
        np.testing.assert_allclose(U5[:, ::-1], U1, atol=1e-10)
        np.testing.assert_allclose(U6[:, ::-1], U1, atol=1e-10)

    def test_fourier_transform(self):
        s = self._rs.uniform(size=(self._G.N, 99, 21))
        s_hat = self._G.gft(s)
        s_star = self._G.igft(s_hat)
        np.testing.assert_allclose(s, s_star)

    def test_gft_windowed_gabor(self):
        self._G.gft_windowed_gabor(self._signal, lambda x: x/(1.-x))

    def test_gft_windowed(self):
        self.assertRaises(NotImplementedError, self._G.gft_windowed,
                          None, self._signal)

    def test_gft_windowed_normalized(self):
        self.assertRaises(NotImplementedError, self._G.gft_windowed_normalized,
                          None, self._signal)

    def test_translate(self):
        self.assertRaises(NotImplementedError, self._G.translate,
                          self._signal, 42)

    def test_modulate(self):
        # FIXME: don't work
        # self._G.modulate(self._signal, 3)
        pass

    def test_edge_list(self):
        G = graphs.StochasticBlockModel(undirected=True)
        v_in, v_out, weights = G.get_edge_list()
        self.assertEqual(G.W[v_in[42], v_out[42]], weights[42])

        G = graphs.StochasticBlockModel(undirected=False)
        self.assertRaises(NotImplementedError, G.get_edge_list)

    def test_differential_operator(self):
        G = graphs.StochasticBlockModel(undirected=True)
        L = G.D.T.dot(G.D)
        np.testing.assert_allclose(L.toarray(), G.L.toarray())

        G = graphs.StochasticBlockModel(undirected=False)
        self.assertRaises(NotImplementedError, G.compute_differential_operator)

    def test_difference(self):
        for lap_type in ['combinatorial', 'normalized']:
            G = graphs.Logo(lap_type=lap_type)
            s_grad = G.grad(self._signal)
            Ls = G.div(s_grad)
            np.testing.assert_allclose(Ls, G.L.dot(self._signal))

    def test_set_coordinates(self):
        G = graphs.FullConnected()
        coords = np.random.uniform(size=(G.N, 2))
        G.set_coordinates(coords)
        G.set_coordinates('ring2D')
        G.set_coordinates('random2D')
        G.set_coordinates('random3D')
        G.set_coordinates('spring')
        G.set_coordinates('spring', dim=3)
        G.set_coordinates('spring', dim=3, pos=G.coords)
        self.assertRaises(AttributeError, G.set_coordinates, 'community2D')
        G = graphs.Community()
        G.set_coordinates('community2D')
        self.assertRaises(ValueError, G.set_coordinates, 'invalid')

    def test_sanitize_signal(self):
        s1 = np.arange(self._G.N)
        s2 = np.reshape(s1, (self._G.N, 1))
        s3 = np.reshape(s1, (self._G.N, 1, 1))
        s4 = np.arange(self._G.N*10).reshape((self._G.N, 10))
        s5 = np.reshape(s4, (self._G.N, 10, 1))
        s1 = self._G.sanitize_signal(s1)
        s2 = self._G.sanitize_signal(s2)
        s3 = self._G.sanitize_signal(s3)
        s4 = self._G.sanitize_signal(s4)
        s5 = self._G.sanitize_signal(s5)
        np.testing.assert_equal(s2, s1)
        np.testing.assert_equal(s3, s1)
        np.testing.assert_equal(s5, s4)
        self.assertRaises(ValueError, self._G.sanitize_signal,
                          np.ones((2, 2, 2, 2)))

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
        k = 6
        G = graphs.RandomRegular(k=k)
        np.testing.assert_equal(G.W.sum(0), k)
        np.testing.assert_equal(G.W.sum(1), k)

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
        graphs.Grid2d(3, 2)
        graphs.Grid2d(3)

    def test_imgpatches(self):
        graphs.ImgPatches(img=self._img, patch_shape=(3, 3))

    def test_grid2dimgpatches(self):
        graphs.Grid2dImgPatches(img=self._img, patch_shape=(3, 3))


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
