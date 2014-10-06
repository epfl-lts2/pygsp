#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the modulename module of the pygsp package.
"""

import sys
import numpy as np
import numpy.testing as nptest
import pygsp

# Use the unittest2 backport on Python 2.6 to profit from the new features.
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dummy(self):
        """
        Dummy test.
        """
        a = np.array([1, 2])
        b = pygsp.module1.dummy(1, a, True)
        nptest.assert_almost_equal(a, b)


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
