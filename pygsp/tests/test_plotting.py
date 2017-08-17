#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the plotting module of the pygsp package.

"""

import unittest

import numpy as np
from skimage import data, img_as_float

from pygsp import graphs


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        self._img = img_as_float(data.camera()[::16, ::16])

    def tearDown(self):
        pass

    def test_is_plottable(self):

        classnames = graphs.__all__

        # Graphs who are not embedded.
        classnames.remove('Graph')
        classnames.remove('BarabasiAlbert')
        classnames.remove('ErdosRenyi')
        classnames.remove('FullConnected')
        classnames.remove('RandomRegular')
        classnames.remove('RandomRing')
        classnames.remove('Ring')  # TODO: should have!
        classnames.remove('StochasticBlockModel')

        Gs = []
        for classname in classnames:
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
            # Check attributes.
            self.assertTrue(hasattr(G, 'coords'))
            self.assertTrue(hasattr(G, 'A'))
            self.assertEqual(G.N, G.coords.shape[0])


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)
