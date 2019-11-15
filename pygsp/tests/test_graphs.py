# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

from __future__ import division

import os
import random

import sys
import unittest

import numpy as np
import scipy.linalg
from scipy import sparse
import networkx as nx
from skimage import data, img_as_float

from pygsp import graphs
from pygsp import filters

import graph_tool as gt
import graph_tool.generation




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
        for attr in ['A', 'd', 'dw', 'lmax', 'U', 'e', 'coherence', 'D']:
            # FIXME: The Laplacian L should be there as well.
            self.assertRaises(AttributeError, setattr, G, attr, None)
            self.assertRaises(AttributeError, delattr, G, attr)

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


    def test_nngraph(self, n_vertices=24):
        """Test all the combinations of metric, kind, backend."""
        Graph = graphs.NNGraph
        data = np.random.RandomState(42).uniform(size=(n_vertices, 3))
        metrics = ['euclidean', 'manhattan', 'max_dist', 'minkowski']
        # Not testing , 'flann', 'nmslib' as they are tested in test_nearest_neighbor
        backends = ['scipy-kdtree', 'scipy-ckdtree'] 

        for metric in metrics:
            for kind in ['knn', 'radius']:
                params = dict(features=data, metric=metric, kind=kind, k=6)
                ref = Graph(backend='scipy-pdist', **params)
                for backend in backends:
                    graph = Graph(**params)
                    np.testing.assert_allclose(graph.W.toarray(),
                                                   ref.W.toarray(), rtol=1e-5)

        # Distances between points on a circle.
        angles = [0, 2 * np.pi / n_vertices]
        points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        distance = np.linalg.norm(points[0] - points[1])
        weight = np.exp(-np.log(2) * distance**2)
        column = np.zeros(n_vertices)
        column[1] = weight
        column[-1] = weight
        adjacency = scipy.linalg.circulant(column)
        data = graphs.Ring(n_vertices).coords
        for kind in ['knn', 'radius']:
            for backend in backends + ['scipy-pdist']:
                graph = Graph(data, kind=kind, k=2, radius=1.01*distance,
                              kernel_width=1, backend=backend)
                np.testing.assert_allclose(graph.W.toarray(), adjacency)

        graph = Graph(data, kind='radius', radius='estimate')
        np.testing.assert_allclose(graph.radius, np.sqrt(8 / n_vertices))
        graph = Graph(data, kind='radius', k=2, radius='estimate-knn')
        np.testing.assert_allclose(graph.radius, distance)

        graph = Graph(data, standardize=True)
        np.testing.assert_allclose(np.mean(graph.coords, axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(np.std(graph.coords, axis=0), 1)

        # Invalid parameters.
        self.assertRaises(ValueError, Graph, np.ones(n_vertices))
        self.assertRaises(ValueError, Graph, np.ones((n_vertices, 3, 4)))
        self.assertRaises(ValueError, Graph, data, metric='invalid')
        self.assertRaises(ValueError, Graph, data, kind='invalid')
        self.assertRaises(ValueError, Graph, data, kernel='invalid')
        self.assertRaises(ValueError, Graph, data, backend='invalid')
        self.assertRaises(ValueError, Graph, data, kind='knn', k=0)
        self.assertRaises(ValueError, Graph, data, kind='knn', k=n_vertices)
        self.assertRaises(ValueError, Graph, data, kind='radius', radius=0)

        # Empty graph.
        if sys.version_info > (3, 4):  # no assertLogs in python 2.7
            for backend in backends + ['scipy-pdist']:
                if backend == 'nmslib':
                    continue  # nmslib doesn't support radius
                with self.assertLogs(level='WARNING'):
                    graph = Graph(data, kind='radius', radius=1e-9,
                                  backend=backend)
                self.assertEqual(graph.n_edges, 0)

        # Backend parameters.
        Graph(data, lap_type='normalized')
        Graph(data, plotting=dict(vertex_size=10))
        Graph(data, backend='flann', algorithm='kmeans')
        Graph(data, backend='nmslib', method='vptree')
        Graph(data, backend='nmslib', index=dict(post=2))
        Graph(data, backend='nmslib', query=dict(efSearch=100))
        for backend in ['scipy-kdtree', 'scipy-ckdtree']:
            Graph(data, backend=backend, eps=1e-2)
            Graph(data, backend=backend, leafsize=9)
        self.assertRaises(ValueError, Graph, data, backend='scipy-pdist', a=0)

        # Kernels.
        for name, kernel in graphs.NNGraph._kernels.items():
            similarity = 0 if name == 'rectangular' else 0.5
            np.testing.assert_allclose(kernel(np.ones(10)), similarity)
            np.testing.assert_allclose(kernel(np.zeros(10)), 1)
        Graph(data, kernel=lambda d: d.min()/d)
        if sys.version_info > (3, 4):  # no assertLogs in python 2.7
            with self.assertLogs(level='WARNING'):
                Graph(data, kernel=lambda d: 1/d)

        # Attributes.
        self.assertEqual(Graph(data, kind='knn').radius, None)
        self.assertEqual(Graph(data, kind='radius').k, None)

    def test_line_graph(self):
        adjacency = [
            [0, 1, 1, 3],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [3, 0, 1, 0],
        ]
        coords = [
            [0, 0],
            [4, 0],
            [4, 2],
            [0, 2],
        ]
        graph = graphs.Graph(adjacency, coords=coords)
        graph = graphs.LineGraph(graph)
        adjacency = [
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 1, 1, 1, 0],
        ]
        coords = [
            [2, 0],
            [2, 1],
            [0, 1],
            [4, 1],
            [2, 2],
        ]
        np.testing.assert_equal(graph.W.toarray(), adjacency)
        np.testing.assert_equal(graph.coords, coords)
        if sys.version_info > (3, 4):  # no assertLogs in python 2.7
            with self.assertLogs(level='WARNING'):
                graphs.LineGraph(graphs.Graph([[0, 2], [2, 0]]))

    def test_subgraph(self, n_vertices=100):
        self._G.set_signal(self._G.coords, 'coords')
        graph = self._G.subgraph(range(n_vertices))
        self.assertEqual(graph.n_vertices, n_vertices)
        self.assertEqual(graph.coords.shape, (n_vertices, 2))
        self.assertEqual(graph.signals['coords'].shape, (n_vertices, 2))
        self.assertIs(graph.lap_type, self._G.lap_type)
        self.assertEqual(graph.plotting, self._G.plotting)

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

    def test_grid2d_diagonals(self):
        value = 0.5
        G = graphs.Grid2d(6, 7, diagonal=value)
        self.assertEqual(G.W[2, 8], value)
        self.assertEqual(G.W[9, 1], value)
        self.assertEqual(G.W[9, 3], value)
        self.assertEqual(G.W[2, 14], 0.0)
        self.assertEqual(G.W[17, 1], 0.0)
        self.assertEqual(G.W[9, 16], 1.0)
        self.assertEqual(G.W[20, 27], 1.0)

    def test_learn_graph(self):
        from pygsp.graphs import LearnedFromSmoothSignals
        from pygsp.utils import distanz
        from pygsp.graphs.learngraph import gsp_compute_graph_learning_theta
        
        # Create a bunch of signals
        n=100
        d = 400
        k = 6
        coords = np.random.RandomState(0).uniform(size=(n, 2))
        G = graphs.NNGraph(coords,k=10, kernel='gaussian')
        G.compute_fourier_basis()
        g = filters.Filter(G,lambda x:1/(1+5*x))
        S = np.random.randn(n,d)
        X = np.squeeze(g.filter(S))

        # Test if large scale and small scale are close
        G_learned1 = LearnedFromSmoothSignals(X, k=k, sparse=False)
        G_learned2 = LearnedFromSmoothSignals(X,k=k, sparse=True)
        assert(np.sum(np.abs(G_learned2.W.todense()-G_learned1.W.todense()))/np.sum(np.abs(G_learned1.W.todense()))<1e-4)

        # Test automatic parameters tuning
        Z = distanz(X.transpose())
        theta, theta_min, theta_max = gsp_compute_graph_learning_theta(Z, k)
        a = 1/theta
        b = a
        G_learned3 = LearnedFromSmoothSignals(X,a=a,b=b, sparse=False)
        assert(np.sum(np.abs(G_learned3.W.todense()-G_learned1.W.todense()))/np.sum(np.abs(G_learned1.W.todense()))<1e-3)

suite_graphs = unittest.TestLoader().loadTestsFromTestCase(TestCase)


class TestImportExport(unittest.TestCase):

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

    @unittest.skipIf(sys.version_info < (3,), 'old graph-tool')
    def test_graphtool_export_import(self):
        # Export to graph tool and reimport to PyGSP directly
        # The exported graph is a simple one without an associated Signal
        g = graphs.Bunny()
        g_gt = g.to_graphtool()
        g2 = graphs.Graph.from_graphtool(g_gt)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

    @unittest.skipIf(sys.version_info < (3,), 'old graph-tool')
    def test_graphtool_multiedge_import(self):
        # Manualy create a graph with multiple edges
        g_gt = gt.Graph()
        g_gt.add_vertex(n=10)
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

    @unittest.skipIf(sys.version_info < (3,), 'old graph-tool')
    def test_graphtool_import_export(self):
        # Import to PyGSP and export again to graph tool directly
        # create a random graphTool graph that does not contain multiple edges and no signal
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
        rs = np.random.RandomState(42)
        signal1 = rs.normal(size=graph.N)
        signal2 = rs.normal(size=graph.N)
        graph.set_signal(signal1, "signal1")
        graph.set_signal(signal2, "signal2")
        graph_nx = graph.to_networkx()
        for i in range(graph.N):
            self.assertEqual(graph_nx.nodes[i]["signal1"], signal1[i])
            self.assertEqual(graph_nx.nodes[i]["signal2"], signal2[i])
        # invalid signal type
        graph = graphs.Path(3)
        graph.set_signal(np.array(['a', 'b', 'c']), 'sig')
        self.assertRaises(ValueError, graph.to_networkx)

    def test_graphtool_signal_export(self):
        g = graphs.Logo()
        rs = np.random.RandomState(42)
        s = rs.normal(size=g.N)
        s2 = rs.normal(size=g.N)
        g.set_signal(s, "signal1")
        g.set_signal(s2, "signal2")
        g_gt = g.to_graphtool()
        # Check the signals on all nodes
        for i, v in enumerate(g_gt.vertices()):
            self.assertEqual(g_gt.vertex_properties["signal1"][v], s[i])
            self.assertEqual(g_gt.vertex_properties["signal2"][v], s2[i])
        # invalid signal type
        graph = graphs.Path(3)
        graph.set_signal(np.array(['a', 'b', 'c']), 'sig')
        self.assertRaises(TypeError, graph.to_graphtool)

    @unittest.skipIf(sys.version_info < (3,), 'old graph-tool')
    def test_graphtool_signal_import(self):
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
        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(2, 5))
        graph_nx.add_edges_from([(3, 4), (2, 4), (3, 5)])
        nx.set_node_attributes(graph_nx, {2: 4, 3: 5, 5: 2.3}, "s")
        graph_pg = graphs.Graph.from_networkx(graph_nx)
        np.testing.assert_allclose(graph_pg.signals["s"], [4, 5, np.nan, 2.3])

    def test_no_weights(self):

        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # NetworkX no weights.
        graph_nx = nx.Graph()
        graph_nx.add_edge(0, 1)
        graph_nx.add_edge(1, 2)
        graph_pg = graphs.Graph.from_networkx(graph_nx)
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

        # NetworkX non-existent weight name.
        graph_nx.edges[(0, 1)]['weight'] = 2
        graph_nx.edges[(1, 2)]['weight'] = 2
        graph_pg = graphs.Graph.from_networkx(graph_nx)
        np.testing.assert_allclose(graph_pg.W.toarray(), 2*adjacency)
        graph_pg = graphs.Graph.from_networkx(graph_nx, weight='unknown')
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

        # Graph-tool no weights.
        graph_gt = gt.Graph(directed=False)
        graph_gt.add_edge(0, 1)
        graph_gt.add_edge(1, 2)
        graph_pg = graphs.Graph.from_graphtool(graph_gt)
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

        # Graph-tool non-existent weight name.
        prop = graph_gt.new_edge_property("double")
        prop[(0, 1)] = 2
        prop[(1, 2)] = 2
        graph_gt.edge_properties["weight"] = prop
        graph_pg = graphs.Graph.from_graphtool(graph_gt)
        np.testing.assert_allclose(graph_pg.W.toarray(), 2*adjacency)
        graph_pg = graphs.Graph.from_graphtool(graph_gt, weight='unknown')
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

    def test_break_join_signals(self):
        """Multi-dim signals are broken on export and joined on import."""
        graph1 = graphs.Sensor(20, seed=42)
        graph1.set_signal(graph1.coords, 'coords')
        # networkx
        graph2 = graph1.to_networkx()
        graph2 = graphs.Graph.from_networkx(graph2)
        np.testing.assert_allclose(graph2.signals['coords'], graph1.coords)
        # graph-tool
        graph2 = graph1.to_graphtool()
        graph2 = graphs.Graph.from_graphtool(graph2)
        np.testing.assert_allclose(graph2.signals['coords'], graph1.coords)
        # save and load (need ordered dicts)
        if sys.version_info >= (3, 6):
            filename = 'graph.graphml'
            graph1.save(filename)
            graph2 = graphs.Graph.load(filename)
            np.testing.assert_allclose(graph2.signals['coords'], graph1.coords)
            os.remove(filename)

    @unittest.skipIf(sys.version_info < (3, 6), 'need ordered dicts')
    def test_save_load(self):

        # TODO: test with multiple graphs and signals
        # * dtypes (float, int, bool) of adjacency and signals
        # * empty graph / isolated nodes

        G1 = graphs.Sensor(seed=42)
        W = G1.W.toarray()
        sig = np.random.RandomState(42).normal(size=G1.N)
        G1.set_signal(sig, 's')

        for fmt in ['graphml', 'gml', 'gexf']:
            for backend in ['networkx', 'graph-tool']:

                if fmt == 'gexf' and backend == 'graph-tool':
                    self.assertRaises(ValueError, G1.save, 'g', fmt, backend)
                    self.assertRaises(ValueError, graphs.Graph.load, 'g', fmt,
                                      backend)
                    os.remove('g')
                    continue

                atol = 1e-5 if fmt == 'gml' and backend == 'graph-tool' else 0

                for filename, fmt in [('graph.' + fmt, None), ('graph', fmt)]:
                    G1.save(filename, fmt, backend)
                    G2 = graphs.Graph.load(filename, fmt, backend)
                    np.testing.assert_allclose(G2.W.toarray(), W, atol=atol)
                    np.testing.assert_allclose(G2.signals['s'], sig, atol=atol)
                    os.remove(filename)

        self.assertRaises(ValueError, graphs.Graph.load, 'g.gml', fmt='?')
        self.assertRaises(ValueError, graphs.Graph.load, 'g.gml', backend='?')
        self.assertRaises(ValueError, G1.save, 'g.gml', fmt='?')
        self.assertRaises(ValueError, G1.save, 'g.gml', backend='?')

    @unittest.skipIf(sys.version_info < (3, 3), 'need unittest.mock')
    def test_import_errors(self):
        from unittest.mock import patch
        graph = graphs.Sensor()
        filename = 'graph.gml'
        with patch.dict(sys.modules, {'networkx': None}):
            self.assertRaises(ImportError, graph.to_networkx)
            self.assertRaises(ImportError, graphs.Graph.from_networkx, None)
            self.assertRaises(ImportError, graph.save, filename,
                              backend='networkx')
            self.assertRaises(ImportError, graphs.Graph.load, filename,
                              backend='networkx')
            graph.save(filename)
            graphs.Graph.load(filename)
        with patch.dict(sys.modules, {'graph_tool': None}):
            self.assertRaises(ImportError, graph.to_graphtool)
            self.assertRaises(ImportError, graphs.Graph.from_graphtool, None)
            self.assertRaises(ImportError, graph.save, filename,
                              backend='graph-tool')
            self.assertRaises(ImportError, graphs.Graph.load, filename,
                              backend='graph-tool')
            graph.save(filename)
            graphs.Graph.load(filename)
        with patch.dict(sys.modules, {'networkx': None, 'graph_tool': None}):
            self.assertRaises(ImportError, graph.save, filename)
            self.assertRaises(ImportError, graphs.Graph.load, filename)
        os.remove(filename)


suite_import_export = unittest.TestLoader().loadTestsFromTestCase(TestImportExport)
suite = unittest.TestSuite([suite_graphs, suite_import_export])

