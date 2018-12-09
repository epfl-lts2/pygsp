#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pygsp package.

"""

import unittest

from pygsp.tests import test_graphs
from pygsp.tests import test_filters
from pygsp.tests import test_utils
from pygsp.tests import test_docstrings
from pygsp.tests import test_plotting

loader = unittest.TestLoader()

suites = []

suites.append(test_graphs.suite)
suites.append(test_filters.suite)
suites.append(test_utils.suite)
suites.append(test_docstrings.suite)
suites.append(test_plotting.suite)  # TODO: can SIGSEGV if not last
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
