#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pygsp package.
"""

import unittest
from pygsp.tests import test_docstrings, test_tutorials
from pygsp.tests import test_graphs, test_filters, test_utils, test_plotting


suites = []

suites.append(test_docstrings.suite)
suites.append(test_tutorials.suite)
suites.append(test_graphs.suite)
suites.append(test_utils.suite)
suites.append(test_filters.suite)
suites.append(test_plotting.suite)

suite = unittest.TestSuite(suites)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
