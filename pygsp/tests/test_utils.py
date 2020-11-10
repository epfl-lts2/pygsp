# -*- coding: utf-8 -*-

"""
Test suite for the utils module of the pygsp package.

"""

import unittest

import numpy as np
from scipy import sparse

from pygsp import graphs, utils


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._rs = np.random.RandomState(42)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_symmetrize(self):
        W = sparse.random(100, 100, random_state=42)
        for method in ['average', 'maximum', 'fill', 'tril', 'triu']:
            # Test that the regular and sparse versions give the same result.
            W1 = utils.symmetrize(W, method=method)
            W2 = utils.symmetrize(W.toarray(), method=method)
            np.testing.assert_equal(W1.toarray(), W2)
        self.assertRaises(ValueError, utils.symmetrize, W, 'sum')

    def test_latlon2xyz(self):
        lat = self._rs.uniform(-np.pi/2, np.pi/2, 100)
        lon = self._rs.uniform(0, 2*np.pi, 100)
        xyz = np.stack(utils.latlon2xyz(lat, lon), axis=1)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1)
        lat2, lon2 = utils.xyz2latlon(*xyz.T)
        np.testing.assert_allclose(lat2, lat)
        np.testing.assert_allclose(lon2, lon)

    def test_xyz2latlon(self):
        xyz = self._rs.uniform(-1, 1, (100, 3))
        xyz /= np.linalg.norm(xyz, axis=1)[:, np.newaxis]
        lat, lon = utils.xyz2latlon(*xyz.T)
        self.assertTrue(np.all(lon >= 0))
        self.assertTrue(np.all(lon < 2*np.pi))
        self.assertTrue(np.all(lat >= -np.pi/2))
        self.assertTrue(np.all(lat <= np.pi/2))
        xyz2 = np.stack(utils.latlon2xyz(lat, lon), axis=1)
        np.testing.assert_allclose(xyz2, xyz)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
