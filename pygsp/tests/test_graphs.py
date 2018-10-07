# -*- coding: utf-8 -*-

"""
Test suite for the graphs module of the pygsp package.

"""

import unittest
import random

import numpy as np
import scipy.linalg
import networkx as nx
import graph_tool as gt
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

    def test_degree(self):
        W = 0.3 * (np.ones((4, 4)) - np.diag(4 * [1]))
        G = graphs.Graph(W)
        A = np.ones(W.shape) - np.diag(np.ones(4))
        np.testing.assert_allclose(G.A.toarray(), A)
        np.testing.assert_allclose(G.d, 3 * np.ones([4]))
        np.testing.assert_allclose(G.dw, 3 * 0.3)

    def test_laplacian(self):
        # TODO: should test correctness.

        G = graphs.StochasticBlockModel(N=100, directed=False)
        self.assertFalse(G.is_directed())
        G.compute_laplacian(lap_type='combinatorial')
        G.compute_laplacian(lap_type='normalized')

        G = graphs.StochasticBlockModel(N=100, directed=True)
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
        G = graphs.StochasticBlockModel(N=100, directed=False)
        v_in, v_out, weights = G.get_edge_list()
        self.assertEqual(G.W[v_in[42], v_out[42]], weights[42])

        G = graphs.StochasticBlockModel(N=100, directed=True)
        self.assertRaises(NotImplementedError, G.get_edge_list)

    def test_differential_operator(self):
        G = graphs.StochasticBlockModel(N=100, directed=False)
        L = G.D.T.dot(G.D)
        np.testing.assert_allclose(L.toarray(), G.L.toarray())

        G = graphs.StochasticBlockModel(N=100, directed=True)
        self.assertRaises(NotImplementedError, G.compute_differential_operator)

    def test_difference(self):
        for lap_type in ['combinatorial', 'normalized']:
            G = graphs.Logo(lap_type=lap_type)
            s_grad = G.grad(self._signal)
            Ls = G.div(s_grad)
            np.testing.assert_allclose(Ls, G.L.dot(self._signal))

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

    def test_community(self):
        graphs.Community()
        graphs.Community(comm_density=0.2)
        graphs.Community(k_neigh=5)
        graphs.Community(N=100, Nc=3, comm_sizes=[20, 50, 30])

    def test_minnesota(self):
        graphs.Minnesota()

    def test_sensor(self):
        graphs.Sensor(regular=True)
        graphs.Sensor(regular=False)
        graphs.Sensor(distributed=True)
        graphs.Sensor(distributed=False)
        graphs.Sensor(connected=True)
        graphs.Sensor(connected=False)

    def test_stochasticblockmodel(self):
        graphs.StochasticBlockModel(N=100, directed=True)
        graphs.StochasticBlockModel(N=100, directed=False)
        graphs.StochasticBlockModel(N=100, self_loops=True)
        graphs.StochasticBlockModel(N=100, self_loops=False)
        graphs.StochasticBlockModel(N=100, connected=True)
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
        graphs.ErdosRenyi(N=100, connected=True, directed=False)
        graphs.ErdosRenyi(N=100, connected=True, directed=True)
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

        self.assertTrue(nx.is_isomorphic(g_nx, g))
        np.testing.assert_array_equal(nx.adjacency_matrix(g_nx).todense(),
                                      nx.adjacency_matrix(g).todense())

    def test_graphtool_export_import(self):
        # Export to graph tool and reimport to PyGSP directly
        # The exported graph is a simple one without an associated Signal
        g = graphs.Bunny()
        g_gt = g.to_graphtool()
        g2 = graphs.Graph.from_graphtool(g_gt)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

    def test_graphtool_multiedge_import(self):
        # Manualy create a graph with multiple edges
        g_gt = gt.Graph()
        g_gt.add_vertex(10)
        # connect edge (3,6) three times
        for i in range(3):
            g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
        g = graphs.Graph.from_graphtool(g_gt)
        self.assertEqual(g.W[3, 6], 3.0)

        # test custom aggregator function
        g2 = graphs.Graph.from_graphtool(g_gt, aggr_fun=np.mean)
        self.assertEqual(g2.W[3, 6], 1.0)

        eprop_double = g_gt.new_edge_property("double")

        # Set the weight of 2 out of the 3 edges. The last one has a default weight of 0
        e = g_gt.edge(3, 6, all_edges=True)
        eprop_double[e[0]] = 8.0
        eprop_double[e[1]] = 1.0

        g_gt.edge_properties["weight"] = eprop_double
        g3 = graphs.Graph.from_graphtool(g_gt, aggr_fun=np.mean)
        self.assertEqual(g3.W[3, 6], 3.0)

    def test_graphtool_import_export(self):
        # Import to PyGSP and export again to graph tool directly
        # create a random graphTool graph that does not contain multiple edges and no signal
        graph_gt = gt.Graph()
        graph_gt.add_vertex(100)

        # insert single random links
        eprop_double = graph_gt.new_edge_property("double")
        for s, t in set(zip(np.random.randint(0, 100, 100),
                        np.random.randint(0, 100, 100))):
            graph_gt.add_edge(graph_gt.vertex(s), graph_gt.vertex(t))

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
        logo = graphs.Logo()
        s = np.random.random(logo.N)
        s2 = np.random.random(logo.N)
        logo.set_signal(s, "signal1")
        logo.set_signal(s2, "signal2")
        logo_nx = logo.to_networkx()
        for i in range(50):
            # Randomly check the signal of 50 nodes to see if they are the same
            rd_node = np.random.randint(logo.N)
            self.assertEqual(logo_nx.node[rd_node]["signal1"], s[rd_node])
            self.assertEqual(logo_nx.node[rd_node]["signal2"], s2[rd_node])

    def test_graphtool_signal_export(self):
        g = graphs.Logo()
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
        g = graphs.Graph.from_graphtool(g_gt, signals_names=["signal"])
        self.assertEqual(g.signals["signal"][0], 5.0)
        self.assertEqual(g.signals["signal"][1], -3.0)
        self.assertEqual(g.signals["signal"][2], 2.4)

    def test_networkx_signal_import(self):
        g_nx = nx.Graph()
        g_nx.add_edge(3, 4)
        g_nx.add_edge(2, 4)
        g_nx.add_edge(3, 5)
        print(list(g_nx.node)[0])
        dic_signal = {
            2: 4.0,
            3: 5.0,
            4: 3.3,
            5: 2.3
        }

        nx.set_node_attributes(g_nx, dic_signal, "signal1")
        g = graphs.Graph.from_networkx(g_nx, signals_name=["signal1"])

        for i, node in enumerate(g_nx.node):
            self.assertEqual(g.signals["signal1"][i],
                             nx.get_node_attributes(g_nx, "signal1")[node])

    def test_save_load(self):
        g = graphs.Bunny()
        g.save("bunny.gml")
        g2 = graphs.Graph.load("bunny.gml")
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())


suite = unittest.TestLoader().loadTestsFromTestCase(TestCaseImportExport)
