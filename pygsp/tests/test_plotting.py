# -*- coding: utf-8 -*-

"""
Test suite for the plotting module of the pygsp package.

"""

import unittest
import os

import numpy as np
from skimage import data, img_as_float

from pygsp import graphs, plotting


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._img = img_as_float(data.camera()[::16, ::16])

    @classmethod
    def tearDownClass(cls):
        pass

    def test_plot_graphs(self):
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
                Xin = np.arange(90).reshape(30, 3)
                Gs.append(Graph(Xin))
            elif classname == 'Grid2dImgPatches':
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


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
