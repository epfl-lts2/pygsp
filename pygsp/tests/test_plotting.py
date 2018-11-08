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

        # Coordinates are not in 2D or 3D.
        COORDS_WRONG_DIM = {'ImgPatches'}

        Gs = []
        for classname in set(graphs.__all__) - COORDS_NO - COORDS_WRONG_DIM:
            Graph = getattr(graphs, classname)

            # Classes who require parameters.
            if classname == 'NNGraph':
                Xin = np.arange(90).reshape(30, 3)
                Gs.append(Graph(Xin))
            elif classname in ['ImgPatches', 'Grid2dImgPatches']:
                Gs.append(Graph(img=self._img, patch_shape=(3, 3)))
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
            self.assertTrue(hasattr(G, 'A'))
            self.assertEqual(G.N, G.coords.shape[0])

            signal = np.arange(G.N) + 0.3

            G.plot(backend='pyqtgraph')
            G.plot(backend='matplotlib')
            G.plot_signal(signal, backend='pyqtgraph')
            G.plot_signal(signal, backend='matplotlib')
            plotting.close_all()

    def test_highlight(self):

        def test(G):
            s = np.arange(G.N)
            G.plot_signal(s, backend='matplotlib', highlight=0)
            G.plot_signal(s, backend='matplotlib', highlight=[0])
            G.plot_signal(s, backend='matplotlib', highlight=[0, 1])

        # Test for 1, 2, and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G.set_coordinates('line1D')
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_index(self):

        def test(G):
            G.plot(backend='matplotlib', index=False)
            G.plot(backend='matplotlib', index=True)

        # Test for 2D and 3D graphs.
        G = graphs.Ring(10)
        test(G)
        G = graphs.Torus(Nv=5)
        test(G)

    def test_signals(self):
        """Test the different kind of parameters that can be passed."""
        G = graphs.Sensor()
        G.plot_signal()
        G.plot_signal(G.dw)
        G.plot_signal(G.dw, list(G.dw))
        G.plot_signal(list(G.dw), G.dw)
        G.plot_signal(G.dw[:, np.newaxis], G.dw[np.newaxis, :])
        G.plot_signal('r', 100)
        G.plot_signal((0.5, 0.5, 0.5, 0.5))
        self.assertRaises(ValueError, G.plot_signal, 10)
        self.assertRaises(ValueError, G.plot_signal, (0.5, 0.5))
        self.assertRaises(ValueError, G.plot_signal, size=[2, 3, 4, 5])
        self.assertRaises(ValueError, G.plot_signal, size=[G.dw, G.dw])


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
