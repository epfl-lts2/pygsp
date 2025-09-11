"""
Test suite for the plotting module of the pygsp package.

"""

import numpy as np
import pytest
from matplotlib import pyplot as plt
from skimage import data, img_as_float

from pygsp import filters, graphs, plotting


@pytest.fixture(scope="module")
def test_image():
    """Test image for graph construction."""
    return img_as_float(data.camera()[::16, ::16])


@pytest.fixture(scope="module")
def filter_graph():
    """Graph for filter testing."""
    graph = graphs.Sensor(20, seed=42)
    graph.compute_fourier_basis()
    return graph


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically close all plots after each test."""
    yield
    plotting.close_all()


class TestGraphs:
    """Tests for graph plotting functionality."""

    def test_all_graphs(self, test_image):
        """
        Plot all graphs which have coordinates.
        With and without signal.
        With both backends.
        """

        # Graphs who are not embedded, i.e., have no coordinates.
        COORDS_NO = {
            "Graph",
            "BarabasiAlbert",
            "ErdosRenyi",
            "FullConnected",
            "RandomRegular",
            "StochasticBlockModel",
        }

        Gs = []
        for classname in dir(graphs):
            if not classname[0].isupper():
                # Not a Graph class but a submodule or private stuff.
                continue
            elif classname in COORDS_NO:
                continue
            elif classname == "ImgPatches":
                # Coordinates are not in 2D or 3D.
                continue

            Graph = getattr(graphs, classname)

            # Classes who require parameters.
            if classname == "NNGraph":
                Xin = np.arange(90).reshape(30, 3)
                Gs.append(Graph(Xin))
            elif classname == "Grid2dImgPatches":
                Gs.append(Graph(img=test_image, patch_shape=(3, 3)))
            elif classname == "LineGraph":
                Gs.append(Graph(graphs.Sensor(20, seed=42)))
            else:
                Gs.append(Graph())

            # Add more test cases.
            if classname == "TwoMoons":
                Gs.append(Graph(moontype="standard"))
                Gs.append(Graph(moontype="synthesized"))
            elif classname == "Cube":
                Gs.append(Graph(nb_dim=2))
                Gs.append(Graph(nb_dim=3))
            elif classname == "DavidSensorNet":
                Gs.append(Graph(N=64))
                Gs.append(Graph(N=500))
                Gs.append(Graph(N=128))

        for G in Gs:
            assert hasattr(G, "coords")
            assert G.N == G.coords.shape[0]

            signal = np.arange(G.N) + 0.3

            G.plot(backend="pyqtgraph")
            G.plot(backend="matplotlib")
            G.plot(signal, backend="pyqtgraph")
            G.plot(signal, backend="matplotlib")
            plotting.close_all()

    def test_highlight(self):
        """Test highlighting functionality."""

        def test(G):
            s = np.arange(G.N)
            G.plot(s, backend="matplotlib", highlight=0)
            G.plot(s, backend="matplotlib", highlight=[0])
            G.plot(s, backend="matplotlib", highlight=[0, 1])

        # Test for 1, 2, and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G.set_coordinates("line1D")
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_indices(self):
        """Test index display functionality."""

        def test(G):
            G.plot(backend="matplotlib", indices=False)
            G.plot(backend="matplotlib", indices=True)

        # Test for 2D and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_signals(self):
        """Test the different kind of signals that can be plotted."""
        G = graphs.Sensor()
        G.plot()
        rng = np.random.default_rng(42)

        def test_color(param, length):
            for value in [
                "r",
                4 * (0.5,),
                length * (2,),
                np.ones([1, length]),
                rng.random(length),
                np.ones([length, 3]),
                ["red"] * length,
                rng.random([length, 4]),
            ]:
                params = {param: value}
                G.plot(**params)

            for value in [
                10,
                (0.5, 0.5),
                np.ones([length, 2]),
                np.ones([2, length, 3]),
                np.ones([length, 3]) * 1.1,
            ]:
                params = {param: value}
                with pytest.raises(ValueError):
                    G.plot(**params)

            for value in ["r", 4 * (0.5)]:
                params = {param: value, "backend": "pyqtgraph"}
                with pytest.raises(ValueError):
                    G.plot(**params)

        test_color("vertex_color", G.n_vertices)
        test_color("edge_color", G.n_edges)

        def test_size(param, length):
            for value in [15, length * (2,), np.ones([1, length]), rng.random(length)]:
                params = {param: value}
                G.plot(**params)

            for value in [(2, 3, 4, 5), np.ones([2, length]), np.ones([2, length, 3])]:
                params = {param: value}
                with pytest.raises(ValueError):
                    G.plot(**params)

        test_size("vertex_size", G.n_vertices)
        test_size("edge_width", G.n_edges)

    def test_show_close(self):
        """Test show and close functionality."""
        G = graphs.Sensor()
        G.plot()
        plotting.show(block=False)  # Don't block or the test will halt.
        plotting.close()
        plotting.close_all()

    def test_coords(self):
        """Test coordinate validation."""
        G = graphs.Sensor()
        del G.coords
        with pytest.raises(AttributeError):
            G.plot()
        G.coords = None
        with pytest.raises(AttributeError):
            G.plot()
        G.coords = np.ones((G.N, 4))
        with pytest.raises(AttributeError):
            G.plot()
        G.coords = np.ones((G.N, 3, 1))
        with pytest.raises(AttributeError):
            G.plot()
        G.coords = np.ones((G.N // 2, 3))
        with pytest.raises(AttributeError):
            G.plot()

    def test_unknown_backend(self):
        """Test unknown backend handling."""
        G = graphs.Sensor()
        with pytest.raises(ValueError):
            G.plot(backend="abc")


class TestFilters:
    """Tests for filter plotting functionality."""

    def test_all_filters(self, filter_graph):
        """Plot all filters."""
        for classname in dir(filters):
            if not classname[0].isupper():
                # Not a Filter class but a submodule or private stuff.
                continue
            Filter = getattr(filters, classname)
            if classname in ["Filter", "Modulation", "Gabor"]:
                g = Filter(filter_graph, filters.Heat(filter_graph))
            else:
                g = Filter(filter_graph)
            g.plot()
            plotting.close_all()

    def test_evaluation_points(self, filter_graph):
        """Change number of evaluation points."""

        def check(ax, n_lines, n_points):
            assert len(ax.lines) == n_lines  # n_filters + sum
            x, y = ax.lines[0].get_data()
            assert len(x) == n_points
            assert len(y) == n_points

        g = filters.Abspline(filter_graph, 5)
        fig, ax = g.plot(eigenvalues=False)
        check(ax, 6, 500)
        fig, ax = g.plot(40, eigenvalues=False)
        check(ax, 6, 40)
        fig, ax = g.plot(n=20, eigenvalues=False)
        check(ax, 6, 20)

    def test_eigenvalues(self):
        """Plot with and without showing the eigenvalues."""
        graph = graphs.Sensor(20, seed=42)
        graph.estimate_lmax()
        filters.Heat(graph).plot()
        filters.Heat(graph).plot(eigenvalues=False)
        graph.compute_fourier_basis()
        filters.Heat(graph).plot()
        filters.Heat(graph).plot(eigenvalues=True)
        filters.Heat(graph).plot(eigenvalues=False)

    def test_sum_and_labels(self, filter_graph):
        """Plot with and without sum or labels."""

        def test(g):
            for sum in [None, True, False]:
                for labels in [None, True, False]:
                    g.plot(sum=sum, labels=labels)

        test(filters.Heat(filter_graph, 10))  # one filter
        test(filters.Heat(filter_graph, [10, 100]))  # multiple filters

    def test_title(self, filter_graph):
        """Check plot title."""
        fig, ax = filters.Wave(filter_graph, 2, 1).plot()
        assert ax.get_title() == "Wave(in=1, out=1, time=[2.00], speed=[1.00])"
        fig, ax = filters.Wave(filter_graph).plot(title="test")
        assert ax.get_title() == "test"

    def test_ax(self, filter_graph):
        """Axes are returned, but automatically created if not passed."""
        fig, ax = plt.subplots()
        fig2, ax2 = filters.Heat(filter_graph).plot(ax=ax)
        assert fig2 is fig
        assert ax2 is ax

    def test_kwargs(self, filter_graph):
        """Additional parameters can be passed to the mpl functions."""
        g = filters.Heat(filter_graph)
        g.plot(alpha=1)
        g.plot(linewidth=2)
        g.plot(linestyle="-")
        g.plot(label="myfilter")
