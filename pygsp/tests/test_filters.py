#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for the filters module of the pygsp package."""

import sys
from pygsp import graphs, filters

# Use the unittest2 backport on Python 2.6 to profit from the new features.
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_filters(self):
        G = graphs.Logo()
        G.estimate_lmax()

        def fu(x):
            x / (1. + x)

        def test_default_filters(G, fu):
            g = filters.Filter(G)
            g1 = filters.Filter(G, filters=fu)

        def test_abspline(G):
            g = filters.Abspline(G, Nf=4)

        def test_expwin(G):
            g = filters.Expwin(G)

        def test_gabor(G, fu):
            g = filters.Gabor(G, fu)

        def test_halfcosine(G):
            g = filters.Halfcosine(G, Nf=4)

        def test_heat(G):
            g = filters.Heat(G)

        def test_held(G):
            g = filters.Held(G)
            g1 = filters.Held(G, a=0.25)

        def test_itersine(G):
            g = filters.itersine(G, Nf=4)

        def test_mexicanhat(G):
            g = filters.Mexicanhat(G, Nf=5)
            g1 = filters.Mexicanhat(G, Nf=4)

        def test_meyer(G):
            g = filters.Meyer(G, Nf=4)

        def test_papadakis(G):
            g = filters.Papadakis(G)
            g1 = filters.Papadakis(G, a=0.25)

        def test_regular(G):
            g = filters.Regular(G)
            g1 = filters.Regular(G, d=5)
            g2 = filters.Regular(G, d=0)

        def test_simoncelli(G):
            g = filters.Simoncelli(G)
            g1 = filters.Simoncelli(G, a=0.25)

        def test_simpletf(G):
            g = filters.Simpletf(G, Nf=4)

        def test_warpedtranslates(G):
            pass
            # gw = filters.warpedtranslates(G, g))

suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
