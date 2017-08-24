#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pygsp package.

"""

import os
import unittest
import doctest

from pygsp.tests import test_graphs, test_filters
from pygsp.tests import test_utils, test_plotting


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, module_relative=False)


suites = []
suites.append(test_graphs.suite)
suites.append(test_filters.suite)
suites.append(test_utils.suite)
suites.append(test_docstrings('pygsp', '.py'))
suites.append(test_docstrings('.', '.rst'))
suites.append(test_plotting.suite)  # TODO: can SIGSEGV if not last
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
