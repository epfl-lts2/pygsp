#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test suite for the filters module of the pygsp package."""

import unittest
from pygsp import graphs, filters
from numpy import zeros


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

        # Warped translates are not implemented yet
        def test_warpedtranslates(G):
            pass
            # gw = filters.warpedtranslates(G, g))

        """
        Test of the different methods implemented for the filter analysis
        Checks if the 'exact', 'cheby' or 'lanczos' produce the same output
        of a Heat kernel on the Logo graph
        """
        def test_analysis(G):
            # Using Kronecker signal at the node 8.3
            S = zeros(G.N)
            vertex_delta = 83
            S[vertex_delta] = 1
            g = filters.Heat(G)
            c_exact = g.analysis(G, S, method='exact')
            c_cheby = g.analysis(G, S, method='cheby')
            # c_lancz = g.analysis(G, S, method='lanczos')
            self.assertAlmostEqual(c_exact, c_cheby)
            # lanczos analysis is not working for now
            # self.assertAlmostEqual(c_exact, c_lanczos)



suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    """Run tests."""
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
