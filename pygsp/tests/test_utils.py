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
        pass

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


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
