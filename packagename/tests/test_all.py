#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the pyunlocbox package.
"""

import unittest
from packagename.tests import test_docstrings, test_tutorials
from packagename.tests import test_module1, test_module2


suites = []

suites.append(test_docstrings.suite)
suites.append(test_tutorials.suite)
suites.append(test_module1.suite)
suites.append(test_module2.suite)

suite = unittest.TestSuite(suites)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
