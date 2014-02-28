#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_filter
----------------------------------

Tests for filter.py
"""

import unittest

import sys
import numpy as np
sys.path.append("../pygsp")
from pygsp import filter


class TestFilter(unittest.TestCase):
    def setUp(self):        
        print "setUp done"

    def test_print(self):
        g = lambda x: np.sin(x)
        f = filter.GraphFilter(g)
        print f

    def test_print_heat(self):
        fheat = filter.HeatGraphFilter(tau=0.9)
        print fheat
    def test_print_gauss(self):
        fheat = filter.GaussianGraphFilter(tau=0.9)
        print fheat
    def test_print_rect(self):
        fheat = filter.RectGraphFilter(tau=0.9)
        print fheat

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
