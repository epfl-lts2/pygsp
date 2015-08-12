#!/usr/bin/env python
# -*- coding: utf-8 -*-

# When importing the tests, you surely want these modules.
from pygsp.tests import test_all
from pygsp.tests import test_docstrings, test_tutorials
from pygsp.tests import test_graphs, test_filters, test_utils, test_plotting

# Silence the code checker warning about unused symbols.
assert test_all
assert test_docstrings
assert test_tutorials
assert test_graphs
assert test_filters
assert test_utils
assert test_plotting
