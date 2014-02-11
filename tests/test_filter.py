#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_filter
----------------------------------

Tests for filter.py
"""

import unittest

import sys
sys.path.append("../pygsp")
from pygsp import filter


class TestFilter(unittest.TestCase):

    def setUp(self):
        self.f = filter.HeatGraphFilter(tau=0.9)

    def test_print(self):
        print self.f

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
