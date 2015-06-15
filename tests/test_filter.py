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

class TestCheby(unittest.TestCase):

    def setUp(self):
        self.f = filter.HeatGraphFilter(tau=0.5)
        self.c_coeffs=filter.compute_cheby_coeff(self.f.kernel, order=5, N=None, arange=(-1.0, 1.0))
        
        self.f0 = filter.HeatGraphFilter(tau=0)
        self.c_coeffs0=filter.compute_cheby_coeff(self.f0.kernel, order=10, N=None, arange=(-1.0, 1.0))
        print round(sum(self.c_coeffs0)-2,14)
        self.assertFalse(round(sum(self.c_coeffs0)-2,14),'Wrong computation of cheby coeffs')

        self.f100 = filter.HeatGraphFilter(tau=100)
        self.c_coeffs100=filter.compute_cheby_coeff(self.f100.kernel, order=10, N=None, arange=(-1.0, 1.0))

        self.f1p = filter.HeatGraphFilter(tau=1)
        self.c_coeffs1p=filter.compute_cheby_coeff(self.f1p.kernel, order=100, N=None, arange=(-1.0, 1.0))


    def test_print(self):
        print 'Cheby coeffs: '
        print self.c_coeffs
        print self.c_coeffs0
        print self.c_coeffs100
        print self.c_coeffs1p

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
