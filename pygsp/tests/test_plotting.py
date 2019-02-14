# -*- coding: utf-8 -*-

"""
Test suite for the plotting module of the pygsp package.

"""

import unittest
import os

import numpy as np
from matplotlib import pyplot as plt
from skimage import data, img_as_float

from pygsp import graphs, filters, plotting


class TestGraphs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._img = img_as_float(data.camera()[::16, ::16])

    def tearDown(cls):
        plotting.close_all()

    def test_all_graphs(self):
        r"""
        Plot all graphs which have coordinates.
        With and without signal.
        With both backends.
        """

        # Graphs who are not embedded, i.e., have no coordinates.
        COORDS_NO = {
            'Graph',
            'BarabasiAlbert',
            'ErdosRenyi',
            'FullConnected',
            'RandomRegular',
            'StochasticBlockModel',
            }

        Gs = []
        for classname in dir(graphs):

            if not classname[0].isupper():
                # Not a Graph class but a submodule or private stuff.
                continue
            elif classname in COORDS_NO:
                continue
            elif classname == 'ImgPatches':
                # Coordinates are not in 2D or 3D.
                continue

            Graph = getattr(graphs, classname)

            # Classes who require parameters.
            if classname == 'NNGraph':
                features = np.random.RandomState(42).normal(size=(30, 3))
                Gs.append(Graph(features))
            elif classname in ['ImgPatches', 'Grid2dImgPatches']:
                Gs.append(Graph(img=self._img, patch_shape=(3, 3)))
            elif classname == 'LineGraph':
                Gs.append(Graph(graphs.Sensor(20, seed=42)))
            else:
                Gs.append(Graph())

            # Add more test cases.
            if classname == 'TwoMoons':
                Gs.append(Graph(moontype='standard'))
                Gs.append(Graph(moontype='synthesized'))
            elif classname == 'Cube':
                Gs.append(Graph(nb_dim=2))
                Gs.append(Graph(nb_dim=3))
            elif classname == 'DavidSensorNet':
                Gs.append(Graph(N=64))
                Gs.append(Graph(N=500))
                Gs.append(Graph(N=128))

        for G in Gs:
            self.assertTrue(hasattr(G, 'coords'))
            self.assertEqual(G.N, G.coords.shape[0])

            signal = np.arange(G.N) + 0.3

            G.plot(backend='pyqtgraph')
            G.plot(backend='matplotlib')
            G.plot(signal, backend='pyqtgraph')
            G.plot(signal, backend='matplotlib')
            plotting.close_all()

    def test_highlight(self):

        def test(G):
            s = np.arange(G.N)
            G.plot(s, backend='matplotlib', highlight=0)
            G.plot(s, backend='matplotlib', highlight=[0])
            G.plot(s, backend='matplotlib', highlight=[0, 1])

        # Test for 1, 2, and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G.set_coordinates('line1D')
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_indices(self):

        def test(G):
            G.plot(backend='matplotlib', indices=False)
            G.plot(backend='matplotlib', indices=True)

        # Test for 2D and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_signals(self):
        """Test the different kind of signals that can be plotted."""
        G = graphs.Sensor()
        G.plot()
        def test_color(param, length):
            for value in ['r', 4*(.5,), length*(2,), np.ones([1, length]),
                          np.random.RandomState(42).uniform(size=length),
                          np.ones([length, 3]), ["red"] * length,
                          np.random.RandomState(42).rand(length, 4)]:
                params = {param: value}
                G.plot(**params)
            for value in [10, (0.5, 0.5), np.ones([length, 2]),
                          np.ones([2, length, 3]),
                          np.ones([length, 3]) * 1.1]:
                params = {param: value}
                self.assertRaises(ValueError, G.plot, **params)
            for value in ['r', 4*(.5)]:
                params = {param: value, 'backend': 'pyqtgraph'}
                self.assertRaises(ValueError, G.plot, **params)
        test_color('vertex_color', G.n_vertices)
        test_color('edge_color', G.n_edges)
        def test_size(param, length):
            for value in [15, length*(2,), np.ones([1, length]),
                          np.random.RandomState(42).uniform(size=length)]:
                params = {param: value}
                G.plot(**params)
            for value in [(2, 3, 4, 5), np.ones([2, length]),
                          np.ones([2, length, 3])]:
                params = {param: value}
                self.assertRaises(ValueError, G.plot, **params)
        test_size('vertex_size', G.n_vertices)
        test_size('edge_width', G.n_edges)

    def test_show_close(self):
        G = graphs.Sensor()
        G.plot()
        plotting.show(block=False)  # Don't block or the test will halt.
        plotting.close()
        plotting.close_all()

    def test_coords(self):
        G = graphs.Sensor()
        del G.coords
        self.assertRaises(AttributeError, G.plot)
        G.coords = None
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N, 4))
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N, 3, 1))
        self.assertRaises(AttributeError, G.plot)
        G.coords = np.ones((G.N//2, 3))
        self.assertRaises(AttributeError, G.plot)

    def test_unknown_backend(self):
        G = graphs.Sensor()
        self.assertRaises(ValueError, G.plot, backend='abc')


class TestFilters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._graph = graphs.Sensor(20, seed=42)
        cls._graph.compute_fourier_basis()

    def tearDown(cls):
        plotting.close_all()

    def test_all_filters(self):
        """Plot all filters."""
        for classname in dir(filters):
            if not classname[0].isupper():
                # Not a Filter class but a submodule or private stuff.
                continue
            Filter = getattr(filters, classname)
            if classname in ['Filter', 'Modulation', 'Gabor']:
                g = Filter(self._graph, filters.Heat(self._graph))
            else:
                g = Filter(self._graph)
            g.plot()
            plotting.close_all()

    def test_evaluation_points(self):
        """Change number of evaluation points."""
        def check(ax, n_lines, n_points):
            self.assertEqual(len(ax.lines), n_lines)  # n_filters + sum
            x, y = ax.lines[0].get_data()
            self.assertEqual(len(x), n_points)
            self.assertEqual(len(y), n_points)
        g = filters.Abspline(self._graph, 5)
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

    def test_sum_and_labels(self):
        """Plot with and without sum or labels."""
        def test(g):
            for sum in [None, True, False]:
                for labels in [None, True, False]:
                    g.plot(sum=sum, labels=labels)
        test(filters.Heat(self._graph, 10))  # one filter
        test(filters.Heat(self._graph, [10, 100]))  # multiple filters

    def test_title(self):
        """Check plot title."""
        fig, ax = filters.Wave(self._graph, 2, 1).plot()
        assert ax.get_title() == 'Wave(in=1, out=1, time=[2.00], speed=[1.00])'
        fig, ax = filters.Wave(self._graph).plot(title='test')
        assert ax.get_title() == 'test'

    def test_ax(self):
        """Axes are returned, but automatically created if not passed."""
        fig, ax = plt.subplots()
        fig2, ax2 = filters.Heat(self._graph).plot(ax=ax)
        self.assertIs(fig2, fig)
        self.assertIs(ax2, ax)

    def test_kwargs(self):
        """Additional parameters can be passed to the mpl functions."""
        g = filters.Heat(self._graph)
        g.plot(alpha=1)
        g.plot(linewidth=2)
        g.plot(linestyle='-')
        g.plot(label='myfilter')


suite = unittest.TestSuite([
    unittest.TestLoader().loadTestsFromTestCase(TestGraphs),
    unittest.TestLoader().loadTestsFromTestCase(TestFilters),
])
