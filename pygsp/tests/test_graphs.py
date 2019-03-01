# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

from __future__ import division

import sys
import unittest
import random
import os

import numpy as np
import scipy.linalg
from scipy import sparse
import networkx as nx
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
        adjacency = [
            [0., 3., 0., 2.],
            [3., 0., 4., 0.],
            [0., 4., 0., 5.],
            [2., 0., 5., 0.],
        ]

        # Input types.
        G = graphs.Graph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = np.array(adjacency)
        G = graphs.Graph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = sparse.coo_matrix(adjacency)
        G = graphs.Graph(adjacency)
        self.assertIs(type(G.W), sparse.csr_matrix)
        adjacency = sparse.csr_matrix(adjacency)
        # G = graphs.Graph(adjacency)
        # self.assertIs(G.W, adjacency)  # Not copied if already CSR.

        # Attributes.
        np.testing.assert_allclose(G.W.toarray(), adjacency.toarray())
        np.testing.assert_allclose(G.A.toarray(), G.W.toarray() > 0)
        np.testing.assert_allclose(G.d, np.array([2, 2, 2, 2]))
        np.testing.assert_allclose(G.dw, np.array([5, 7, 9, 7]))
        self.assertEqual(G.n_vertices, 4)
        self.assertIs(G.N, G.n_vertices)
        self.assertEqual(G.n_edges, 4)
        self.assertIs(G.Ne, G.n_edges)

        # Errors and warnings.
        self.assertRaises(ValueError, graphs.Graph, np.ones((3, 4)))
        self.assertRaises(ValueError, graphs.Graph, np.ones((3, 3, 4)))
        self.assertRaises(ValueError, graphs.Graph, [[0, np.nan], [0, 0]])
        self.assertRaises(ValueError, graphs.Graph, [[0, np.inf], [0, 0]])
        if sys.version_info > (3, 4):  # no assertLogs in python 2.7
            with self.assertLogs(level='WARNING'):
                graphs.Graph([[0, -1], [-1, 0]])
            with self.assertLogs(level='WARNING'):
                graphs.Graph([[1, 1], [1, 0]])

    def test_degree(self):
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        np.testing.assert_allclose(graph.d, [1, 2, 1])
        np.testing.assert_allclose(graph.dw, [1, 3, 2])
        graph = graphs.Graph([
            [0, 1, 0],
            [0, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        np.testing.assert_allclose(graph.d, [0.5, 1.5, 1])
        np.testing.assert_allclose(graph.dw, [0.5, 2.5, 2])

    def test_is_connected(self):
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), True)
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), False)
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), False)
        graph = graphs.Graph([
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), True)

    def test_is_directed(self):
        graph = graphs.Graph([
            [0, 3, 0, 0],
            [3, 0, 4, 0],
            [0, 4, 0, 2],
            [0, 0, 2, 0],
        ])
        assert graph.W.nnz == 6
        self.assertEqual(graph.is_directed(), False)
        # In-place modification is not allowed anymore.
        # graph.W[0, 1] = 0
        # assert graph.W.nnz == 6
        # self.assertEqual(graph.is_directed(recompute=True), True)
        # graph.W[1, 0] = 0
        # assert graph.W.nnz == 6
        # self.assertEqual(graph.is_directed(recompute=True), False)

    def test_is_connected(self):
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), True)
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 2, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), False)
        graph = graphs.Graph([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), False)
        self.assertEqual(graph.is_connected(), False)
        graph = graphs.Graph([
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ])
        self.assertEqual(graph.is_directed(), True)
        self.assertEqual(graph.is_connected(), True)

    def test_is_directed(self):
        graph = graphs.Graph([
            [0, 3, 0, 0],
            [3, 0, 4, 0],
            [0, 4, 0, 2],
            [0, 0, 2, 0],
        ])
        assert graph.W.nnz == 6
        self.assertEqual(graph.is_directed(), False)
        graph.W[0, 1] = 0
        assert graph.W.nnz == 6
        self.assertEqual(graph.is_directed(recompute=True), True)
        graph.W[1, 0] = 0
        assert graph.W.nnz == 6
        self.assertEqual(graph.is_directed(recompute=True), False)

    def test_laplacian(self):

        G = graphs.Graph([
            [0, 3, 0, 1],
            [3, 0, 1, 0],
            [0, 1, 0, 3],
            [1, 0, 3, 0],
        ])
        laplacian = np.array([
            [+4, -3, +0, -1],
            [-3, +4, -1, +0],
            [+0, -1, +4, -3],
            [-1, +0, -3, +4],
        ])
        self.assertFalse(G.is_directed())
        G.compute_laplacian('combinatorial')
        np.testing.assert_allclose(G.L.toarray(), laplacian)
        G.compute_laplacian('normalized')
        np.testing.assert_allclose(G.L.toarray(), laplacian/4)

        G = graphs.Graph([
            [0, 6, 0, 1],
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [1, 0, 3, 0],
        ])
        self.assertTrue(G.is_directed())
        G.compute_laplacian('combinatorial')
        np.testing.assert_allclose(G.L.toarray(), laplacian)
        G.compute_laplacian('normalized')
        np.testing.assert_allclose(G.L.toarray(), laplacian/4)

        def test_combinatorial(G):
            np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
            np.testing.assert_equal(G.L.sum(axis=0), 0)
            np.testing.assert_equal(G.L.sum(axis=1), 0)
            np.testing.assert_equal(G.L.diagonal(), G.dw)

        def test_normalized(G):
            np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
            np.testing.assert_equal(G.L.diagonal(), 1)

        G = graphs.ErdosRenyi(100, directed=False)
        self.assertFalse(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        test_combinatorial(G)
        G.compute_laplacian(lap_type='normalized')
        test_normalized(G)

        G = graphs.ErdosRenyi(100, directed=True)
        self.assertTrue(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        test_combinatorial(G)
        G.compute_laplacian(lap_type='normalized')
        test_normalized(G)

    def test_estimate_lmax(self):

        graph = graphs.Sensor()
        self.assertRaises(ValueError, graph.estimate_lmax, method='unk')

        def check_lmax(graph, lmax):
            graph.estimate_lmax(method='bounds')
            np.testing.assert_allclose(graph.lmax, lmax)
            graph.estimate_lmax(method='lanczos')
            np.testing.assert_allclose(graph.lmax, lmax*1.01)
            graph.compute_fourier_basis()
            np.testing.assert_allclose(graph.lmax, lmax)

        # Full graph (bound is tight).
        n_nodes, value = 10, 2
        adjacency = np.full((n_nodes, n_nodes), value)
        graph = graphs.Graph(adjacency, lap_type='combinatorial')
        check_lmax(graph, lmax=value*n_nodes)

        # Regular bipartite graph (bound is tight).
        adjacency = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
        graph = graphs.Graph(adjacency, lap_type='combinatorial')
        check_lmax(graph, lmax=4)

        # Bipartite graph (bound is tight).
        adjacency = [
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        graph = graphs.Graph(adjacency, lap_type='normalized')
        check_lmax(graph, lmax=2)

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
        n = G.N // 2
        # check partial eigendecomposition
        G.compute_fourier_basis(n_eigenvectors=n)
        assert len(G.e) == n
        assert G.U.shape[1] == n
        assert G.e[-1] < 2
        U = G.U
        e = G.e
        # check full eigendecomposition
        G.compute_fourier_basis()
        assert len(G.e) == G.N
        assert G.U.shape[1] == G.N
        assert G.e[-1] < 2
        # eigsh might flip a sign
        np.testing.assert_allclose(np.abs(U), np.abs(G.U[:, :n]),
                                   atol=1e-12)
        np.testing.assert_allclose(e, G.e[:n])

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

    def test_edge_list(self):
        for directed in [False, True]:
            G = graphs.ErdosRenyi(100, directed=directed)
            sources, targets, weights = G.get_edge_list()
            if not directed:
                self.assertTrue(np.all(sources <= targets))
            edges = np.arange(G.n_edges)
            np.testing.assert_equal(G.W[sources[edges], targets[edges]],
                                    weights[edges][np.newaxis, :])

    def test_differential_operator(self, n_vertices=98):
        r"""The Laplacian must always be the divergence of the gradient,
        whether the Laplacian is combinatorial or normalized, and whether the
        graph is directed or weighted."""
        def test_incidence_nx(graph):
            r"""Test that the incidence matrix corresponds to NetworkX."""
            incidence_pg = np.sign(graph.D.toarray())
            G = nx.OrderedDiGraph if graph.is_directed() else nx.OrderedGraph
            graph_nx = nx.from_scipy_sparse_matrix(graph.W, create_using=G)
            incidence_nx = nx.incidence_matrix(graph_nx, oriented=True)
            np.testing.assert_equal(incidence_pg, incidence_nx.toarray())
        for graph in [graphs.Graph(np.zeros((n_vertices, n_vertices))),
                      graphs.Graph(np.identity(n_vertices)),
                      graphs.Graph([[0, 0.8], [0.8, 0]]),
                      graphs.Graph([[1.3, 0], [0.4, 0.5]]),
                      graphs.ErdosRenyi(n_vertices, directed=False, seed=42),
                      graphs.ErdosRenyi(n_vertices, directed=True, seed=42)]:
            for lap_type in ['combinatorial', 'normalized']:
                graph.compute_laplacian(lap_type)
                graph.compute_differential_operator()
                L = graph.D.dot(graph.D.T)
                np.testing.assert_allclose(L.toarray(), graph.L.toarray())
                test_incidence_nx(graph)

    def test_difference(self):
        for lap_type in ['combinatorial', 'normalized']:
            G = graphs.Logo(lap_type=lap_type)
            y = G.grad(self._signal)
            self.assertEqual(len(y), G.n_edges)
            z = G.div(y)
            self.assertEqual(len(z), G.n_vertices)
            np.testing.assert_allclose(z, G.L.dot(self._signal))

    def test_dirichlet_energy(self, n_vertices=100):
        r"""The Dirichlet energy is defined as the norm of the gradient."""
        signal = np.random.RandomState(42).uniform(size=n_vertices)
        for lap_type in ['combinatorial', 'normalized']:
            graph = graphs.BarabasiAlbert(n_vertices)
            graph.compute_differential_operator()
            energy = graph.dirichlet_energy(signal)
            grad_norm = np.sum(graph.grad(signal)**2)
            np.testing.assert_allclose(energy, grad_norm)

    def test_empty_graph(self, n_vertices=11):
        """Empty graphs have either no edge, or self-loops only. The Laplacian
        doesn't see self-loops, as the gradient on those edges is always zero.
        """
        adjacencies = [
            np.zeros((n_vertices, n_vertices)),
            np.identity(n_vertices),
        ]
        for adjacency, n_edges in zip(adjacencies, [0, n_vertices]):
            graph = graphs.Graph(adjacency)
            self.assertEqual(graph.n_vertices, n_vertices)
            self.assertEqual(graph.n_edges, n_edges)
            self.assertEqual(graph.W.nnz, n_edges)
            for laplacian in ['combinatorial', 'normalized']:
                graph.compute_laplacian(laplacian)
                self.assertEqual(graph.L.nnz, 0)
                sources, targets, weights = graph.get_edge_list()
                self.assertEqual(len(sources), n_edges)
                self.assertEqual(len(targets), n_edges)
                self.assertEqual(len(weights), n_edges)
                graph.compute_differential_operator()
                self.assertEqual(graph.D.nnz, 0)
                graph.compute_fourier_basis()
                np.testing.assert_allclose(graph.U, np.identity(n_vertices))
                np.testing.assert_allclose(graph.e, np.zeros(n_vertices))
            # NetworkX uses the same conventions.
            G = nx.from_scipy_sparse_matrix(graph.W)
            self.assertEqual(nx.laplacian_matrix(G).nnz, 0)
            self.assertEqual(nx.normalized_laplacian_matrix(G).nnz, 0)
            self.assertEqual(nx.incidence_matrix(G).nnz, 0)

    def test_adjacency_types(self, n_vertices=10):

        rs = np.random.RandomState(42)
        W = 10 * np.abs(rs.normal(size=(n_vertices, n_vertices)))
        W = W + W.T
        W = W - np.diag(np.diag(W))

        def test(adjacency):
            G = graphs.Graph(adjacency)
            G.compute_laplacian('combinatorial')
            G.compute_laplacian('normalized')
            G.estimate_lmax()
            G.compute_fourier_basis()
            G.compute_differential_operator()

        test(W)
        test(W.astype(np.float32))
        test(W.astype(np.int))
        test(sparse.csr_matrix(W))
        test(sparse.csr_matrix(W, dtype=np.float32))
        test(sparse.csr_matrix(W, dtype=np.int))
        test(sparse.csc_matrix(W))
        test(sparse.coo_matrix(W))

    def test_set_signal(self, name='test'):
        signal = np.zeros(self._G.n_vertices)
        self._G.set_signal(signal, name)
        self.assertIs(self._G.signals[name], signal)
        signal = np.zeros(self._G.n_vertices // 2)
        self.assertRaises(ValueError, self._G.set_signal, signal, name)

    def test_set_coordinates(self):
        G = graphs.FullConnected()
        coords = self._rs.uniform(size=(G.N, 2))
        G.set_coordinates(coords)
        G.set_coordinates('ring2D')
        G.set_coordinates('random2D')
        G.set_coordinates('random3D')
        G.set_coordinates('spring')
        G.set_coordinates('spring', dim=3)
        G.set_coordinates('spring', dim=3, pos=G.coords)
        G.set_coordinates('laplacian_eigenmap2D')
        G.set_coordinates('laplacian_eigenmap3D')
        self.assertRaises(AttributeError, G.set_coordinates, 'community2D')
        G = graphs.Community()
        G.set_coordinates('community2D')
        self.assertRaises(ValueError, G.set_coordinates, 'invalid')


    def test_subgraph(self, n_vertices=100):
        graph = self._G.subgraph(range(n_vertices))
        self.assertEqual(graph.n_vertices, n_vertices)
        self.assertEqual(graph.coords.shape, (n_vertices, 2))
        self.assertIs(graph.lap_type, self._G.lap_type)
        self.assertEqual(graph.plotting, self._G.plotting)

    def test_nngraph(self, n_vertices=30):
        rs = np.random.RandomState(42)
        Xin = rs.normal(size=(n_vertices, 3))
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
        graphs.TwoMoons(moontype='standard')
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
        self.assertRaises(ValueError, graphs.Ring, 2)
        self.assertRaises(ValueError, graphs.Ring, 5, k=3)

    def test_community(self):
        graphs.Community()
        graphs.Community(comm_density=0.2)
        graphs.Community(k_neigh=5)
        graphs.Community(N=100, Nc=3, comm_sizes=[20, 50, 30])

    def test_minnesota(self):
        graphs.Minnesota()

    def test_sensor(self):
        graphs.Sensor(3000)
        graphs.Sensor(N=100, distributed=True)
        self.assertRaises(ValueError, graphs.Sensor, N=101, distributed=True)
        graphs.Sensor(N=101, distributed=False)
        graphs.Sensor(seed=10)
        graphs.Sensor(k=20)

    def test_stochasticblockmodel(self):
        graphs.StochasticBlockModel(N=100, directed=True)
        graphs.StochasticBlockModel(N=100, directed=False)
        graphs.StochasticBlockModel(N=100, self_loops=True)
        graphs.StochasticBlockModel(N=100, self_loops=False)
        graphs.StochasticBlockModel(N=100, connected=True, seed=42)
        graphs.StochasticBlockModel(N=100, connected=False)
        self.assertRaises(ValueError, graphs.StochasticBlockModel,
                          N=100, p=0, q=0, connected=True)

    def test_airfoil(self):
        graphs.Airfoil()

    def test_davidsensornet(self):
        graphs.DavidSensorNet()
        graphs.DavidSensorNet(N=500)
        graphs.DavidSensorNet(N=128)

    def test_erdosreny(self):
        graphs.ErdosRenyi(N=100, connected=False, directed=False)
        graphs.ErdosRenyi(N=100, connected=False, directed=True)
        graphs.ErdosRenyi(N=100, connected=True, directed=False, seed=42)
        graphs.ErdosRenyi(N=100, connected=True, directed=True, seed=42)
        G = graphs.ErdosRenyi(N=100, p=1, self_loops=True)
        self.assertEqual(G.W.nnz, 100**2)

    def test_fullconnected(self):
        graphs.FullConnected()

    def test_logo(self):
        graphs.Logo()

    def test_path(self):
        graphs.Path()

    def test_randomring(self):
        graphs.RandomRing()
        G = graphs.RandomRing(angles=[0, 2, 1])
        self.assertEqual(G.N, 3)
        self.assertRaises(ValueError, graphs.RandomRing, 2)
        self.assertRaises(ValueError, graphs.RandomRing, angles=[0, 2])
        self.assertRaises(ValueError, graphs.RandomRing, angles=[0, 2, 7])
        self.assertRaises(ValueError, graphs.RandomRing, angles=[0, 2, -1])

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


suite_graphs = unittest.TestLoader().loadTestsFromTestCase(TestCase)


class TestCaseImportExport(unittest.TestCase):

    def test_networkx_export_import(self):
        # Export to networkx and reimport to PyGSP

        # Exporting the Bunny graph
        g = graphs.Bunny()
        g_nx = g.to_networkx()
        g2 = graphs.Graph.from_networkx(g_nx)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

    def test_networkx_import_export(self):
        # Import from networkx then export to networkx again
        g_nx = nx.gnm_random_graph(100, 50)  # Generate a random graph
        g = graphs.Graph.from_networkx(g_nx).to_networkx()

        np.testing.assert_array_equal(nx.adjacency_matrix(g_nx).todense(),
                                      nx.adjacency_matrix(g).todense())

    def test_graphtool_export_import(self):
        # Export to graph tool and reimport to PyGSP directly
        # The exported graph is a simple one without an associated Signal
        if sys.version_info < (3, 0):
            return None  # skip test for python 2.7
        g = graphs.Bunny()
        g_gt = g.to_graphtool()
        g2 = graphs.Graph.from_graphtool(g_gt)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

    def test_graphtool_multiedge_import(self):
        if sys.version_info < (3, 0):
            return None  # skip test for python2.7
        # Manualy create a graph with multiple edges
        g_gt = gt.Graph()
        g_gt.add_vertex(10)
        # connect edge (3,6) three times
        for i in range(3):
            g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
        g = graphs.Graph.from_graphtool(g_gt)
        self.assertEqual(g.W[3, 6], 3.0)

        eprop_double = g_gt.new_edge_property("double")

        # Set the weight of 2 out of the 3 edges. The last one has a default weight of 0
        e = g_gt.edge(3, 6, all_edges=True)
        eprop_double[e[0]] = 8.0
        eprop_double[e[1]] = 1.0

        g_gt.edge_properties["weight"] = eprop_double
        g3 = graphs.Graph.from_graphtool(g_gt)
        self.assertEqual(g3.W[3, 6], 9.0)

    def test_graphtool_import_export(self):
        # Import to PyGSP and export again to graph tool directly
        # create a random graphTool graph that does not contain multiple edges and no signal
        if sys.version_info < (3, 0):
            return None  # skip test for python2.7
        graph_gt = gt.generation.random_graph(100, lambda : (np.random.poisson(4), np.random.poisson(4)))

        eprop_double = graph_gt.new_edge_property("double")
        for e in graph_gt.edges():
            eprop_double[e] = random.random()
        graph_gt.edge_properties["weight"] = eprop_double

        graph2_gt = graphs.Graph.from_graphtool(graph_gt).to_graphtool()

        self.assertEqual(graph_gt.num_edges(), graph2_gt.num_edges(),
                         "the number of edges does not correspond")

        def key(edge): return str(edge.source()) + ":" + str(edge.target())

        for e1, e2 in zip(sorted(graph_gt.edges(), key=key), sorted(graph2_gt.edges(), key=key)):
            self.assertEqual(e1.source(), e2.source())
            self.assertEqual(e1.target(), e2.target())
        for v1, v2 in zip(graph_gt.vertices(), graph2_gt.vertices()):
            self.assertEqual(v1, v2)

    def test_networkx_signal_export(self):
        graph = graphs.BarabasiAlbert(N=100, seed=42)
        np.random.seed(42)
        signal1 = np.random.random(graph.N)
        signal2 = np.random.random(graph.N)
        graph.set_signal(signal1, "signal1")
        graph.set_signal(signal2, "signal2")
        graph_nx = graph.to_networkx()
        for i in range(graph.N):
            self.assertEqual(graph_nx.node[i]["signal1"], signal1[i])
            self.assertEqual(graph_nx.node[i]["signal2"], signal2[i])

    def test_graphtool_signal_export(self):
        g = graphs.Logo()
        np.random.seed(42)
        s = np.random.random(g.N)
        s2 = np.random.random(g.N)
        g.set_signal(s, "signal1")
        g.set_signal(s2, "signal2")
        g_gt = g.to_graphtool()
        # Check the signals on all nodes
        for i, v in enumerate(g_gt.vertices()):
            self.assertEqual(g_gt.vertex_properties["signal1"][v], s[i])
            self.assertEqual(g_gt.vertex_properties["signal2"][v], s2[i])

    def test_graphtool_signal_import(self):
        if sys.version_info < (3, 0):
            return None  # skip test for python2.7
        g_gt = gt.Graph()
        g_gt.add_vertex(10)

        g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
        g_gt.add_edge(g_gt.vertex(4), g_gt.vertex(6))
        g_gt.add_edge(g_gt.vertex(7), g_gt.vertex(2))

        vprop_double = g_gt.new_vertex_property("double")

        vprop_double[g_gt.vertex(0)] = 5
        vprop_double[g_gt.vertex(1)] = -3
        vprop_double[g_gt.vertex(2)] = 2.4

        g_gt.vertex_properties["signal"] = vprop_double
        g = graphs.Graph.from_graphtool(g_gt)
        self.assertEqual(g.signals["signal"][0], 5.0)
        self.assertEqual(g.signals["signal"][1], -3.0)
        self.assertEqual(g.signals["signal"][2], 2.4)

    def test_networkx_signal_import(self):
        g_nx = nx.Graph()
        g_nx.add_edge(3, 4)
        g_nx.add_edge(2, 4)
        g_nx.add_edge(3, 5)
        dic_signal = {
            2: 4.0,
            3: 5.0,
            4: 3.3,
            5: 2.3
        }

        nx.set_node_attributes(g_nx, dic_signal, "signal1")
        g = graphs.Graph.from_networkx(g_nx)

        for i, node in enumerate(g_nx.node):
            self.assertEqual(g.signals["signal1"][i],
                             nx.get_node_attributes(g_nx, "signal1")[node])

    def test_save_load(self):
        if sys.version_info >= (3, 6):
            graph = graphs.Sensor(seed=42)
            np.random.seed(42)
            signal = np.random.random(graph.N)
            graph.set_signal(signal, "signal")

            # save
            nx_gt = ['gml', 'graphml']
            all_files = []
            for fmt in nx_gt:
                all_files += ["graph_gt.{}".format(fmt), "graph_nx.{}".format(fmt)]
                graph.save("graph_gt.{}".format(fmt), backend='graph_tool')
                graph.save("graph_nx.{}".format(fmt), backend='networkx')
            graph.save("graph_nx.{}".format('gexf'), backend='networkx')
            all_files += ["graph_nx.{}".format('gexf')]

            # load
            for filename in all_files:
                if not "_gt" in filename:
                    graph_loaded_nx = graphs.Graph.load(filename, backend='networkx')
                    np.testing.assert_array_equal(graph.W.todense(), graph_loaded_nx.W.todense())
                    np.testing.assert_array_equal(signal, graph_loaded_nx.signals['signal'])
                if not ".gexf" in filename:
                    graph_loaded_gt = graphs.Graph.load(filename, backend='graph_tool')
                    np.testing.assert_allclose(graph.W.todense(), graph_loaded_gt.W.todense(), atol=0.000001)
                    np.testing.assert_allclose(signal, graph_loaded_gt.signals['signal'], atol=0.000001)

            # clean
            for filename in all_files:
                os.remove(filename)


suite_import_export = unittest.TestLoader().loadTestsFromTestCase(TestCaseImportExport)
suite = unittest.TestSuite([suite_graphs, suite_import_export])
