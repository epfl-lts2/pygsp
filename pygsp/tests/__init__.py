#!/usr/bin/env python
# -*- coding: utf-8 -*-

# When importing the tests, you surely want these modules.
from pygsp.tests import test_all
from pygsp.tests import test_docstrings, test_tutorials
from pygsp.tests import test_module1, test_module2

# Silence the code checker warning about unused symbols.
assert test_all
assert test_docstrings
assert test_tutorials
assert test_module1
assert test_module2
