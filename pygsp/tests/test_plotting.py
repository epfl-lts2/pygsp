#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the plotting module of the pygsp package.

"""

import unittest

import numpy as np
from skimage import data, img_as_float

from pygsp import graphs, plotting


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self._img = img_as_float(data.camera()[::16, ::16])

    def tearDown(self):
        pass

    def test_plot_graphs(self):
        r"""
        Plot all graphs which have coordinates.
        With and without signal.
        With both backends.
        """

        # Graphs who are not embedded, i.e. have no coordinates.
        COORDS_NO = {
            'Graph',
            'BarabasiAlbert',
            'ErdosRenyi',
            'FullConnected',
            'RandomRegular',
            'RandomRing',
            'Ring',  # TODO: should have!
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

            if G.is_directed():
                self.assertRaises(NotImplementedError,
                                  G.plot, default_qtg=True)
                self.assertRaises(NotImplementedError,
                                  G.plot, default_qtg=False)
            else:
                # Backend: pyqtgraph.
                G.plot(default_qtg=True)
                G.plot_signal(signal, default_qtg=True)
                # Backend: matplotlib.
                G.plot(default_qtg=False)
                G.plot_signal(signal, default_qtg=False)
                plotting.close_all()


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)
