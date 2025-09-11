"""
Test suite for the graphs module of the pygsp package.

"""

import os
import random
import sys

import networkx as nx
import numpy as np
import pytest
import scipy.linalg
from scipy import sparse
from skimage import data, img_as_float

try:
    import graph_tool as gt
    import graph_tool.generation

    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    GRAPH_TOOL_AVAILABLE = False

from pygsp import graphs

N = 123


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def test_graph():
    """Graph for filter testing."""
    G = graphs.Sensor(N, seed=42)
    G.compute_fourier_basis()
    G.compute_differential_operator()
    return G


@pytest.fixture(scope="module")
def graph_signal(rng):
    """Graph signal for testing."""
    return rng.uniform(size=N)


@pytest.fixture(scope="module")
def test_image():
    """Test image for testing."""
    return img_as_float(data.camera()[::16, ::16])


def test_graph_input_types(caplog):
    adjacency = [
        [0.0, 3.0, 0.0, 2.0],
        [3.0, 0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0, 5.0],
        [2.0, 0.0, 5.0, 0.0],
    ]

    # Input types.
    G = graphs.Graph(adjacency)
    assert isinstance(G.W, sparse.csr_matrix)
    adjacency = np.array(adjacency)
    G = graphs.Graph(adjacency)
    assert isinstance(G.W, sparse.csr_matrix)
    adjacency = sparse.coo_matrix(adjacency)
    G = graphs.Graph(adjacency)
    assert isinstance(G.W, sparse.csr_matrix)
    adjacency = sparse.csr_matrix(adjacency)
    # G = graphs.Graph(adjacency)
    # self.assertIs(G.W, adjacency)  # Not copied if already CSR.

    # Attributes.
    np.testing.assert_allclose(G.W.toarray(), adjacency.toarray())
    np.testing.assert_allclose(G.A.toarray(), G.W.toarray() > 0)
    np.testing.assert_allclose(G.d, np.array([2, 2, 2, 2]))
    np.testing.assert_allclose(G.dw, np.array([5, 7, 9, 7]))
    assert G.n_vertices == 4
    assert G.N == G.n_vertices
    assert G.n_edges == 4
    assert G.Ne == G.n_edges

    # Errors and warnings.
    with pytest.raises(ValueError):
        graphs.Graph(np.ones((3, 4)))
    with pytest.raises(ValueError):
        graphs.Graph(np.ones((3, 3, 4)))
    with pytest.raises(ValueError):
        graphs.Graph([[0, np.nan], [0, 0]])
    with pytest.raises(ValueError):
        graphs.Graph([[0, np.inf], [0, 0]])
    with caplog.at_level("WARNING"):
        graphs.Graph([[0, -1], [-1, 0]])
    assert len(caplog.records) > 0
    caplog.clear()
    with caplog.at_level("WARNING"):
        graphs.Graph([[1, 1], [1, 0]])
    assert len(caplog.records) > 0
    for attr in ["A", "d", "dw", "lmax", "U", "e", "coherence", "D"]:
        # FIXME: The Laplacian L should be there as well.
        with pytest.raises(AttributeError):
            setattr(G, attr, None)
        with pytest.raises(AttributeError):
            delattr(G, attr)
        with pytest.raises(AttributeError):
            delattr(G, attr)


def test_degree():
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ]
    )
    assert not graph.is_directed()
    np.testing.assert_allclose(graph.d, [1, 2, 1])
    np.testing.assert_allclose(graph.dw, [1, 3, 2])
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [0, 0, 2],
            [0, 2, 0],
        ]
    )
    assert graph.is_directed()
    np.testing.assert_allclose(graph.d, [0.5, 1.5, 1])
    np.testing.assert_allclose(graph.dw, [0.5, 2.5, 2])


def test_is_connected():
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ]
    )
    assert not graph.is_directed()
    assert graph.is_connected()
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 2, 0],
        ]
    )
    assert graph.is_directed()
    assert not graph.is_connected()
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )
    assert not graph.is_directed()
    assert not graph.is_connected()
    graph = graphs.Graph(
        [
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ]
    )
    assert graph.is_directed()
    assert graph.is_connected()


def test_is_directed():
    graph = graphs.Graph(
        [
            [0, 3, 0, 0],
            [3, 0, 4, 0],
            [0, 4, 0, 2],
            [0, 0, 2, 0],
        ]
    )
    assert graph.W.nnz == 6
    assert not graph.is_directed()
    # In-place modification is not allowed anymore.
    # graph.W[0, 1] = 0
    # assert graph.W.nnz == 6
    # self.assertEqual(graph.is_directed(recompute=True), True)
    # graph.W[1, 0] = 0
    # assert graph.W.nnz == 6
    # self.assertEqual(graph.is_directed(recompute=True), False)


def test_laplacian():
    G = graphs.Graph(
        [
            [0, 3, 0, 1],
            [3, 0, 1, 0],
            [0, 1, 0, 3],
            [1, 0, 3, 0],
        ]
    )
    laplacian = np.array(
        [
            [+4, -3, +0, -1],
            [-3, +4, -1, +0],
            [+0, -1, +4, -3],
            [-1, +0, -3, +4],
        ]
    )
    assert not G.is_directed()
    G.compute_laplacian("combinatorial")
    np.testing.assert_allclose(G.L.toarray(), laplacian)
    G.compute_laplacian("normalized")
    np.testing.assert_allclose(G.L.toarray(), laplacian / 4)

    G = graphs.Graph(
        [
            [0, 6, 0, 1],
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [1, 0, 3, 0],
        ]
    )
    assert G.is_directed()
    G.compute_laplacian("combinatorial")
    np.testing.assert_allclose(G.L.toarray(), laplacian)
    G.compute_laplacian("normalized")
    np.testing.assert_allclose(G.L.toarray(), laplacian / 4)

    def test_combinatorial(G):
        np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
        np.testing.assert_equal(G.L.sum(axis=0), 0)
        np.testing.assert_equal(G.L.sum(axis=1), 0)
        np.testing.assert_equal(G.L.diagonal(), G.dw)

    def test_normalized(G):
        np.testing.assert_equal(G.L.toarray(), G.L.T.toarray())
        np.testing.assert_equal(G.L.diagonal(), 1)

    G = graphs.ErdosRenyi(100, directed=False)
    assert not G.is_directed()
    G.compute_laplacian(lap_type="combinatorial")
    test_combinatorial(G)
    G.compute_laplacian(lap_type="normalized")
    test_normalized(G)

    G = graphs.ErdosRenyi(100, directed=True)
    assert G.is_directed()
    G.compute_laplacian(lap_type="combinatorial")
    test_combinatorial(G)
    G.compute_laplacian(lap_type="normalized")
    test_normalized(G)


def test_estimate_lmax():
    graph = graphs.Sensor()
    with pytest.raises(ValueError):
        graph.estimate_lmax(method="unk")

    def check_lmax(graph, lmax):
        graph.estimate_lmax(method="bounds")
        np.testing.assert_allclose(graph.lmax, lmax)
        graph.estimate_lmax(method="lanczos")
        np.testing.assert_allclose(graph.lmax, lmax * 1.01)
        graph.compute_fourier_basis()
        np.testing.assert_allclose(graph.lmax, lmax)

    # Full graph (bound is tight).
    n_nodes, value = 10, 2
    adjacency = np.full((n_nodes, n_nodes), value)
    graph = graphs.Graph(adjacency, lap_type="combinatorial")
    check_lmax(graph, lmax=value * n_nodes)

    # Regular bipartite graph (bound is tight).
    adjacency = [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ]
    graph = graphs.Graph(adjacency, lap_type="combinatorial")
    check_lmax(graph, lmax=4)

    # Bipartite graph (bound is tight).
    adjacency = [
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    graph = graphs.Graph(adjacency, lap_type="normalized")
    check_lmax(graph, lmax=2)


def test_fourier_basis(test_graph):
    # Smallest eigenvalue close to zero.
    np.testing.assert_allclose(test_graph.e[0], 0, atol=1e-12)
    # First eigenvector is constant.
    N = test_graph.N
    np.testing.assert_allclose(test_graph.U[:, 0], np.sqrt(N) / N)
    # Control eigenvector direction.
    # assert (self._G.U[0, :] > 0).all()
    # Spectrum bounded by [0, 2] for the normalized Laplacian.
    G = graphs.Logo(lap_type="normalized")
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
    np.testing.assert_allclose(np.abs(U), np.abs(G.U[:, :n]), atol=1e-12)
    np.testing.assert_allclose(e, G.e[:n])


def test_eigendecompositions():
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


def test_fourier_transform(rng, test_graph, graph_signal):
    s = rng.uniform(size=(test_graph.N, 99, 21))
    s_hat = test_graph.gft(s)
    s_star = test_graph.igft(s_hat)
    np.testing.assert_allclose(s, s_star, rtol=1e-6)


def test_edge_list():
    for directed in [False, True]:
        G = graphs.ErdosRenyi(100, directed=directed)
        sources, targets, weights = G.get_edge_list()
        if not directed:
            assert np.all(sources <= targets)
        edges = np.arange(G.n_edges)
        np.testing.assert_equal(
            G.W[sources[edges], targets[edges]], weights[edges][np.newaxis, :]
        )


def test_differential_operator(n_vertices=98):
    r"""The Laplacian must always be the divergence of the gradient,
    whether the Laplacian is combinatorial or normalized, and whether the
    graph is directed or weighted."""

    def test_incidence_nx(graph):
        r"""Test that the incidence matrix corresponds to NetworkX."""
        incidence_pg = np.sign(graph.D.toarray())
        G = nx.DiGraph if graph.is_directed() else nx.Graph
        graph_nx = nx.from_scipy_sparse_array(graph.W, create_using=G)
        incidence_nx = nx.incidence_matrix(graph_nx, oriented=True)
        np.testing.assert_equal(incidence_pg, incidence_nx.toarray())

    for graph in [
        graphs.Graph(np.zeros((n_vertices, n_vertices))),
        graphs.Graph(np.identity(n_vertices)),
        graphs.Graph([[0, 0.8], [0.8, 0]]),
        graphs.Graph([[1.3, 0], [0.4, 0.5]]),
        graphs.ErdosRenyi(n_vertices, directed=False, seed=42),
        graphs.ErdosRenyi(n_vertices, directed=True, seed=42),
    ]:
        for lap_type in ["combinatorial", "normalized"]:
            graph.compute_laplacian(lap_type)
            graph.compute_differential_operator()
            L = graph.D.dot(graph.D.T)
            np.testing.assert_allclose(L.toarray(), graph.L.toarray())
            test_incidence_nx(graph)


def test_difference(graph_signal, test_graph):
    for lap_type in ["combinatorial", "normalized"]:
        y = test_graph.grad(graph_signal)
        assert len(y) == test_graph.n_edges
        z = test_graph.div(y)
        assert len(z) == test_graph.n_vertices
        np.testing.assert_allclose(z, test_graph.L.dot(graph_signal))


def test_dirichlet_energy(rng, n_vertices=100):
    r"""The Dirichlet energy is defined as the norm of the gradient."""
    signal = rng.uniform(size=n_vertices)
    for lap_type in ["combinatorial", "normalized"]:
        graph = graphs.BarabasiAlbert(n_vertices, lap_type=lap_type)
        graph.compute_differential_operator()
        energy = graph.dirichlet_energy(signal)
        grad_norm = np.sum(graph.grad(signal) ** 2)
        np.testing.assert_allclose(energy, grad_norm)


def test_empty_graph(n_vertices=11):
    """Empty graphs have either no edge, or self-loops only. The Laplacian
    doesn't see self-loops, as the gradient on those edges is always zero.
    """
    adjacencies = [
        np.zeros((n_vertices, n_vertices)),
        np.identity(n_vertices),
    ]
    for adjacency, n_edges in zip(adjacencies, [0, n_vertices]):
        graph = graphs.Graph(adjacency)
        assert graph.n_vertices == n_vertices
        assert graph.n_edges == n_edges
        assert graph.W.nnz == n_edges
        for laplacian in ["combinatorial", "normalized"]:
            graph.compute_laplacian(laplacian)
            assert graph.L.nnz == 0
            sources, targets, weights = graph.get_edge_list()
            assert len(sources) == n_edges
            assert len(targets) == n_edges
            assert len(weights) == n_edges
            graph.compute_differential_operator()
            assert graph.D.nnz == 0
            graph.compute_fourier_basis()
            np.testing.assert_allclose(graph.U, np.identity(n_vertices))
            np.testing.assert_allclose(graph.e, np.zeros(n_vertices))
        # NetworkX uses the same conventions.
        G = nx.from_scipy_sparse_array(graph.W)
        assert nx.laplacian_matrix(G).nnz == 0
        assert nx.normalized_laplacian_matrix(G).nnz == 0
        assert nx.incidence_matrix(G).nnz == 0


def test_adjacency_types(n_vertices=10):
    rng = np.random.default_rng(42)
    W = 10 * np.abs(rng.normal(size=(n_vertices, n_vertices)))
    W = W + W.T
    W = W - np.diag(np.diag(W))

    def test(adjacency):
        G = graphs.Graph(adjacency)
        G.compute_laplacian("combinatorial")
        G.compute_laplacian("normalized")
        G.estimate_lmax()
        G.compute_fourier_basis()
        G.compute_differential_operator()

    test(W)
    test(W.astype(np.float32))
    test(W.astype(int))
    test(sparse.csr_matrix(W))
    test(sparse.csr_matrix(W, dtype=np.float32))
    test(sparse.csr_matrix(W, dtype=int))
    test(sparse.csc_matrix(W))
    test(sparse.coo_matrix(W))


def test_set_signal(test_graph, name="test"):
    signal = np.zeros(test_graph.n_vertices)
    test_graph.set_signal(signal, name)
    np.testing.assert_equal(test_graph.signals[name], signal)
    signal = np.zeros(test_graph.n_vertices // 2)
    with pytest.raises(ValueError):
        test_graph.set_signal(signal, name)


def test_set_coordinates(rng):
    G = graphs.FullConnected()
    coords = rng.uniform(size=(G.N, 2))
    G.set_coordinates(coords)
    G.set_coordinates("ring2D")
    G.set_coordinates("random2D")
    G.set_coordinates("random3D")
    G.set_coordinates("spring")
    G.set_coordinates("spring", dim=3)
    G.set_coordinates("spring", dim=3, pos=G.coords)
    G.set_coordinates("laplacian_eigenmap2D")
    G.set_coordinates("laplacian_eigenmap3D")
    with pytest.raises(AttributeError):
        G.set_coordinates("community2D")
    G = graphs.Community()
    G.set_coordinates("community2D")
    with pytest.raises(ValueError):
        G.set_coordinates("invalid")


def test_line_graph(caplog):
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
    with caplog.at_level("WARNING"):
        graphs.LineGraph(graphs.Graph([[0, 2], [2, 0]]))
    assert len(caplog.records) > 0


def test_subgraph(test_graph, n_vertices=100):
    assert n_vertices < test_graph.n_vertices
    test_graph.set_signal(test_graph.coords, "coords")
    graph = test_graph.subgraph(range(n_vertices))
    assert graph.n_vertices == n_vertices
    assert graph.coords.shape == (n_vertices, 2)
    assert graph.signals["coords"].shape == (n_vertices, 2)
    assert graph.lap_type == test_graph.lap_type
    np.testing.assert_equal(graph.plotting, test_graph.plotting)


def test_nngraph(n_vertices=30):
    rng = np.random.default_rng(42)
    Xin = rng.normal(size=(n_vertices, 3))
    dist_types = ["euclidean", "manhattan", "max_dist", "minkowski"]

    for dist_type in dist_types:
        # Only p-norms with 1<=p<=infinity permitted.
        if dist_type != "minkowski":
            graphs.NNGraph(Xin, NNtype="radius", dist_type=dist_type, epsilon=0.1)
            graphs.NNGraph(Xin, NNtype="knn", dist_type=dist_type)

        # Distance type unsupported in the C bindings,
        # use the C++ bindings instead.
        if dist_type != "max_dist":
            graphs.NNGraph(Xin, use_flann=True, NNtype="knn", dist_type=dist_type)


def test_bunny():
    graphs.Bunny()


def test_cube():
    graphs.Cube()
    graphs.Cube(nb_dim=2)


def test_sphere():
    graphs.Sphere()


def test_twomoons():
    graphs.TwoMoons(moontype="standard")
    graphs.TwoMoons(moontype="synthesized")


def test_torus():
    graphs.Torus()


def test_comet(n=100, k=10):
    graph = graphs.Comet(n, k)
    assert graph.n_vertices == n
    assert graph.n_edges == n - 1
    assert graph.dw[0] == k
    graph = graphs.Comet(7, 4)
    adjacency = [
        [0, 1, 1, 1, 1, 0, 0],  # center
        [1, 0, 0, 0, 0, 0, 0],  # branch
        [1, 0, 0, 0, 0, 0, 0],  # branch
        [1, 0, 0, 0, 0, 0, 0],  # branch
        [1, 0, 0, 0, 0, 1, 0],  # tail
        [0, 0, 0, 0, 1, 0, 1],  # tail
        [0, 0, 0, 0, 0, 1, 0],  # tail
    ]
    np.testing.assert_array_equal(graph.W.toarray(), adjacency)
    coords = [[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0], [2, 0], [3, 0]]
    np.testing.assert_allclose(graph.coords, coords, atol=1e-10)
    # Comet generalizes Path.
    g1 = graphs.Comet(n, 0)
    g2 = graphs.Path(n)
    np.testing.assert_array_equal(g1.W.toarray(), g2.W.toarray())
    # Comet generalizes Star.
    g1 = graphs.Comet(n, n - 1)
    g2 = graphs.Star(n)
    np.testing.assert_array_equal(g1.W.toarray(), g2.W.toarray())


def test_star(n=20):
    graph = graphs.Star(n)
    assert graph.n_vertices == n
    assert graph.n_edges == n - 1
    np.testing.assert_array_equal(graph.d, [n - 1] + (n - 1) * [1])
    np.testing.assert_allclose(np.linalg.norm(graph.coords[1:], axis=1), 1)


def test_lowstretchtree():
    graphs.LowStretchTree()


def test_randomregular():
    k = 6
    G = graphs.RandomRegular(k=k, seed=42)
    np.testing.assert_equal(G.W.sum(0), k)
    np.testing.assert_equal(G.W.sum(1), k)


def test_ring():
    graphs.Ring()
    graphs.Ring(N=32, k=16)
    with pytest.raises(ValueError):
        graphs.Ring(2)
    with pytest.raises(ValueError):
        graphs.Ring(5, k=3)


def test_community():
    graphs.Community()
    graphs.Community(comm_density=0.2)
    graphs.Community(k_neigh=5)
    graphs.Community(N=100, Nc=3, comm_sizes=[20, 50, 30])


def test_minnesota():
    graphs.Minnesota()


def test_sensor():
    graphs.Sensor(3000)
    graphs.Sensor(N=100, distributed=True)
    with pytest.raises(ValueError):
        graphs.Sensor(N=101, distributed=True)
    graphs.Sensor(N=101, distributed=False)
    graphs.Sensor(seed=10)
    graphs.Sensor(k=20)


def test_stochasticblockmodel():
    graphs.StochasticBlockModel(N=100, directed=True)
    graphs.StochasticBlockModel(N=100, directed=False)
    graphs.StochasticBlockModel(N=100, self_loops=True)
    graphs.StochasticBlockModel(N=100, self_loops=False)
    graphs.StochasticBlockModel(N=100, connected=True, seed=42)
    graphs.StochasticBlockModel(N=100, connected=False)
    with pytest.raises(ValueError):
        graphs.StochasticBlockModel(N=100, p=0, q=0, connected=True)


def test_airfoil():
    graphs.Airfoil()


def test_davidsensornet():
    graphs.DavidSensorNet()
    graphs.DavidSensorNet(N=500)
    graphs.DavidSensorNet(N=128)


def test_erdosreny():
    graphs.ErdosRenyi(N=100, connected=False, directed=False)
    graphs.ErdosRenyi(N=100, connected=False, directed=True)
    graphs.ErdosRenyi(N=100, connected=True, directed=False, seed=42)
    graphs.ErdosRenyi(N=100, connected=True, directed=True, seed=42)
    G = graphs.ErdosRenyi(N=100, p=1, self_loops=True)
    assert G.W.nnz == 100**2


def test_fullconnected():
    graphs.FullConnected()


def test_logo():
    graphs.Logo()


def test_path(n=5):
    graph = graphs.Path(n, directed=False)
    adjacency = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ]
    coords = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    np.testing.assert_array_equal(graph.W.toarray(), adjacency)
    np.testing.assert_array_equal(graph.coords, coords)
    graph = graphs.Path(n, directed=True)
    adjacency = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    np.testing.assert_array_equal(graph.W.toarray(), adjacency)
    np.testing.assert_array_equal(graph.coords, coords)


def test_randomring():
    graphs.RandomRing()
    G = graphs.RandomRing(angles=[0, 2, 1])
    assert G.N == 3
    with pytest.raises(ValueError):
        graphs.RandomRing(2)
    with pytest.raises(ValueError):
        graphs.RandomRing(angles=[0, 2])
    with pytest.raises(ValueError):
        graphs.RandomRing(angles=[0, 2, 7])
    with pytest.raises(ValueError):
        graphs.RandomRing(angles=[0, 2, -1])


def test_swissroll():
    graphs.SwissRoll(srtype="uniform")
    graphs.SwissRoll(srtype="classic")
    graphs.SwissRoll(noise=True)
    graphs.SwissRoll(noise=False)
    graphs.SwissRoll(dim=2)
    graphs.SwissRoll(dim=3)


def test_grid2d():
    graphs.Grid2d(3, 2)
    graphs.Grid2d(3)


def test_imgpatches(test_image):
    graphs.ImgPatches(img=test_image, patch_shape=(3, 3))


def test_grid2dimgpatches(test_image):
    graphs.Grid2dImgPatches(img=test_image, patch_shape=(3, 3))


def test_grid2d_diagonals():
    value = 0.5
    G = graphs.Grid2d(6, 7, diagonal=value)
    assert G.W[2, 8] == value
    assert G.W[9, 1] == value
    assert G.W[9, 3] == value
    assert G.W[2, 14] == 0.0
    assert G.W[17, 1] == 0.0
    assert G.W[9, 16] == 1.0
    assert G.W[20, 27] == 1.0


def test_networkx_export_import():
    # Export to networkx and reimport to PyGSP

    # Exporting the Bunny graph
    g = graphs.Bunny()
    g_nx = g.to_networkx()
    g2 = graphs.Graph.from_networkx(g_nx)
    np.testing.assert_array_equal(g.W.todense(), g2.W.todense())


def test_networkx_import_export():
    # Import from networkx then export to networkx again
    g_nx = nx.gnm_random_graph(100, 50)  # Generate a random graph
    g = graphs.Graph.from_networkx(g_nx).to_networkx()

    np.testing.assert_array_equal(
        nx.adjacency_matrix(g_nx).todense(), nx.adjacency_matrix(g).todense()
    )


@pytest.mark.skipif(not GRAPH_TOOL_AVAILABLE, reason="graph-tool not available")
def test_graphtool_export_import():
    # Export to graph tool and reimport to PyGSP directly
    # The exported graph is a simple one without an associated Signal
    g = graphs.Bunny()
    g_gt = g.to_graphtool()
    g2 = graphs.Graph.from_graphtool(g_gt)
    np.testing.assert_array_equal(g.W.todense(), g2.W.todense())


@pytest.mark.skipif(not GRAPH_TOOL_AVAILABLE, reason="graph-tool not available")
def test_graphtool_multiedge_import():
    # Manualy create a graph with multiple edges
    g_gt = gt.Graph()
    g_gt.add_vertex(n=10)
    # connect edge (3,6) three times
    for i in range(3):
        g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
    g = graphs.Graph.from_graphtool(g_gt)
    assert g.W[3, 6] == 3.0

    eprop_double = g_gt.new_edge_property("double")

    # Set the weight of 2 out of the 3 edges. The last one has a default weight of 0
    e = g_gt.edge(3, 6, all_edges=True)
    eprop_double[e[0]] = 8.0
    eprop_double[e[1]] = 1.0

    g_gt.edge_properties["weight"] = eprop_double
    g3 = graphs.Graph.from_graphtool(g_gt)
    assert g3.W[3, 6] == 9.0


@pytest.mark.skipif(not GRAPH_TOOL_AVAILABLE, reason="graph-tool not available")
def test_graphtool_import_export(rng):
    # Import to PyGSP and export again to graph tool directly
    # create a random graphTool graph that does not contain multiple edges and no signal
    graph_gt = gt.generation.random_graph(100, lambda: (rng.poisson(4), rng.poisson(4)))

    eprop_double = graph_gt.new_edge_property("double")
    for e in graph_gt.edges():
        eprop_double[e] = random.random()
    graph_gt.edge_properties["weight"] = eprop_double

    graph2_gt = graphs.Graph.from_graphtool(graph_gt).to_graphtool()

    assert (
        graph_gt.num_edges() == graph2_gt.num_edges()
    ), "the number of edges does not correspond"

    def key(edge):
        return str(edge.source()) + ":" + str(edge.target())

    for e1, e2 in zip(
        sorted(graph_gt.edges(), key=key), sorted(graph2_gt.edges(), key=key)
    ):
        assert e1.source() == e2.source()
        assert e1.target() == e2.target()
    for v1, v2 in zip(graph_gt.vertices(), graph2_gt.vertices()):
        assert v1 == v2


def test_networkx_signal_export(rng):
    graph = graphs.BarabasiAlbert(N=100, seed=42)
    signal1 = rng.normal(0, 1, graph.N)
    signal2 = rng.integers(0, 10, graph.N)
    graph.set_signal(signal1, "signal1")
    graph.set_signal(signal2, "signal2")
    graph_nx = graph.to_networkx()
    for i in range(graph.N):
        assert graph_nx.nodes[i]["signal1"] == signal1[i]
        assert graph_nx.nodes[i]["signal2"] == signal2[i]
    # invalid signal type
    graph = graphs.Path(3)
    graph.set_signal(np.array(["a", "b", "c"]), "sig")
    with pytest.raises(ValueError):
        graph.to_networkx()


@pytest.mark.skipif(not GRAPH_TOOL_AVAILABLE, reason="graph-tool not available")
def test_graphtool_signal_export():
    g = graphs.Logo()
    rng = np.random.default_rng(42)
    s = rng.normal(0, 1, size=g.N)
    s2 = rng.integers(0, 10, size=g.N)
    g.set_signal(s, "signal1")
    g.set_signal(s2, "signal2")
    g_gt = g.to_graphtool()
    # Check the signals on all nodes
    for i, v in enumerate(g_gt.vertices()):
        assert g_gt.vertex_properties["signal1"][v] == s[i]
        assert g_gt.vertex_properties["signal2"][v] == s2[i]
    # invalid signal type
    graph = graphs.Path(3)
    graph.set_signal(np.array(["a", "b", "c"]), "sig")
    with pytest.raises(TypeError):
        graph.to_graphtool()


@pytest.mark.skipif(not GRAPH_TOOL_AVAILABLE, reason="graph-tool not available")
def test_graphtool_signal_import():
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
    assert g.signals["signal"][0] == 5.0
    assert g.signals["signal"][1] == -3.0
    assert g.signals["signal"][2] == 2.4


def test_networkx_signal_import():
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(2, 5))
    graph_nx.add_edges_from([(3, 4), (2, 4), (3, 5)])
    nx.set_node_attributes(graph_nx, {2: 4, 3: 5, 5: 2.3}, "s")
    graph_pg = graphs.Graph.from_networkx(graph_nx)
    np.testing.assert_allclose(graph_pg.signals["s"], [4, 5, np.nan, 2.3])


def test_no_weights():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # NetworkX no weights.
    graph_nx = nx.Graph()
    graph_nx.add_edge(0, 1)
    graph_nx.add_edge(1, 2)
    graph_pg = graphs.Graph.from_networkx(graph_nx)
    np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

    # NetworkX non-existent weight name.
    graph_nx.edges[(0, 1)]["weight"] = 2
    graph_nx.edges[(1, 2)]["weight"] = 2
    graph_pg = graphs.Graph.from_networkx(graph_nx)
    np.testing.assert_allclose(graph_pg.W.toarray(), 2 * adjacency)
    graph_pg = graphs.Graph.from_networkx(graph_nx, weight="unknown")
    np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

    # Graph-tool no weights (only if available).
    if GRAPH_TOOL_AVAILABLE:
        graph_gt = gt.Graph(directed=False)
        graph_gt.add_edge(0, 1)
        graph_gt.add_edge(1, 2)
        graph_pg = graphs.Graph.from_graphtool(graph_gt)
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)

    # Graph-tool non-existent weight name (only if available).
    if GRAPH_TOOL_AVAILABLE:
        prop = graph_gt.new_edge_property("double")
        prop[graph_gt.edge(0, 1)] = 2
        prop[graph_gt.edge(1, 2)] = 2
        graph_gt.edge_properties["weight"] = prop
        graph_pg = graphs.Graph.from_graphtool(graph_gt)
        np.testing.assert_allclose(graph_pg.W.toarray(), 2 * adjacency)
        graph_pg = graphs.Graph.from_graphtool(graph_gt, weight="unknown")
        np.testing.assert_allclose(graph_pg.W.toarray(), adjacency)


def test_break_join_signals(tmp_path):
    """Multi-dim signals are broken on export and joined on import."""
    graph1 = graphs.Sensor(20, seed=42)
    graph1.set_signal(graph1.coords, "coords")
    # networkx
    graph2 = graph1.to_networkx()
    graph2 = graphs.Graph.from_networkx(graph2)
    np.testing.assert_allclose(graph2.signals["coords"], graph1.coords)
    # graph-tool (only if available)
    if GRAPH_TOOL_AVAILABLE:
        graph2 = graph1.to_graphtool()
        graph2 = graphs.Graph.from_graphtool(graph2)
        np.testing.assert_allclose(graph2.signals["coords"], graph1.coords)
    # save and load (need ordered dicts)
    filename = str(tmp_path / "graph.graphml")
    graph1.save(filename)
    graph2 = graphs.Graph.load(filename)
    np.testing.assert_allclose(graph2.signals["coords"], graph1.coords)
    os.remove(filename)


def test_save_load(tmp_path):
    # TODO: test with multiple graphs and signals
    # * dtypes (float, int, bool) of adjacency and signals
    # * empty graph / isolated nodes

    # Determine available backends
    backends = ["networkx"]  # networkx is always available in dev dependencies
    if GRAPH_TOOL_AVAILABLE:
        backends.append("graph-tool")

    G1 = graphs.Sensor(seed=42)
    W = G1.W.toarray()
    sig = np.random.default_rng(42).normal(size=G1.N)
    G1.set_signal(sig, "s")

    for backend in backends:
        for fmt in ["graphml", "gml", "gexf"]:
            if fmt == "gexf" and backend == "graph-tool":
                filename = str(tmp_path / ("graph." + fmt))
                with pytest.raises(ValueError):
                    G1.save(filename, fmt, backend)
                with pytest.raises(ValueError):
                    graphs.Graph.load(filename, fmt, backend)
                os.remove(filename)
                continue

            atol = 1e-5 if fmt == "gml" and backend == "graph-tool" else 0

            for filename, fmt in [
                (str(tmp_path / ("graph." + fmt)), None),
                (str(tmp_path / "graph"), fmt),
            ]:
                G1.save(filename, fmt, backend)
                G2 = graphs.Graph.load(filename, fmt, backend)
                np.testing.assert_allclose(G2.W.toarray(), W, atol=atol)
                np.testing.assert_allclose(G2.signals["s"], sig, atol=atol)

    with pytest.raises(ValueError):
        graphs.Graph.load("g.gml", fmt="?")
    with pytest.raises(ValueError):
        graphs.Graph.load("g.gml", backend="?")
    with pytest.raises(ValueError):
        G1.save("g.gml", fmt="?")
    with pytest.raises(ValueError):
        G1.save("g.gml", backend="?")


def test_import_errors(tmp_path):
    from unittest.mock import patch

    graph = graphs.Sensor()
    filename = str(tmp_path / "graph.gml")

    # Only test backends that are actually available
    # This test should verify error handling, not require unavailable backends

    # Test NetworkX-specific errors (always available in dev dependencies)
    with patch.dict(sys.modules, {"networkx": None}):
        with pytest.raises(ImportError):
            graph.to_networkx()
        with pytest.raises(ImportError):
            graphs.Graph.from_networkx(None)
        with pytest.raises(ImportError):
            graph.save(filename, backend="networkx")
        with pytest.raises(ImportError):
            graphs.Graph.load(filename, backend="networkx")

    # Test graph-tool specific errors (only if graph-tool would be available)
    if GRAPH_TOOL_AVAILABLE:
        with patch.dict(sys.modules, {"graph_tool": None}):
            with pytest.raises(ImportError):
                graph.to_graphtool()
            with pytest.raises(ImportError):
                graphs.Graph.from_graphtool(None)
            with pytest.raises(ImportError):
                graph.save(filename, backend="graph-tool")
            with pytest.raises(ImportError):
                graphs.Graph.load(filename, backend="graph-tool")
            # Should fallback to networkx
            graph.save(filename)
            graphs.Graph.load(filename)
            os.remove(filename)

    # Test the case where no backends are available
    with patch.dict(sys.modules, {"networkx": None, "graph_tool": None}):
        with pytest.raises(ImportError):
            graph.save(filename)
        with pytest.raises(ImportError):
            graphs.Graph.load(filename)
