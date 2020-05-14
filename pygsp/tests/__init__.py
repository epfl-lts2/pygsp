# -*- coding: utf-8 -*-

"""
Test suite of the PyGSP package, broken by modules.

"""

import unittest

from . import test_graphs
from . import test_filters
from . import test_utils
from . import test_learning
from . import test_docstrings
from . import test_plotting


suite = unittest.TestSuite([
    test_graphs.suite,
    test_filters.suite,
    test_utils.suite,
    test_learning.suite,
    test_docstrings.suite,
    test_plotting.suite,  # TODO: can SIGSEGV if not last
])
