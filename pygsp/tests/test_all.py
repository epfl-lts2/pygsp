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


def test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, module_relative=False)


def setup(doctest):
    import numpy
    import pygsp
    doctest.globs = {
        'graphs': pygsp.graphs,
        'filters': pygsp.filters,
        'utils': pygsp.utils,
        'np': numpy,
    }

loader = unittest.TestLoader()

suites = []
suites.append(loader.loadTestsFromModule(test_graphs))
suites.append(loader.loadTestsFromModule(test_filters))
suites.append(loader.loadTestsFromModule(test_utils))
suites.append(test_docstrings('pygsp', '.py', setup))
suites.append(test_docstrings('.', '.rst'))  # No setup to not forget imports.
# TODO: can SIGSEGV if not last.
suites.append(loader.loadTestsFromModule(test_plotting))
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    run()
