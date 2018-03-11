# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class Chebyshev(Filter):
    r"""Approximate continuous filters with Chebyshev polynomials.

    Math which explains the polynomial filters sum_k theta_k lambda^k
    Weighted sum of diffused versions of the signal
    Note recursive computation. O(N) computational cost and 4N space.

    Math to show how the coefficients are computed

    Parameters
    ----------
    G : graph
    filters : Filter or array-like
        Either a :class:`Filter` object or a set of Chebyshev coefficients
        represented as an array of size K x F, where K is the polynomial
        order and F the number of filters.
    order : int
        Polynomial order.

    """

    def __init__(self, G, filters, order=30):

        self.G = G
        self.order = order

        try:
            self._compute_coefficients(filters)
            self.Nf = filters.Nf
        except:
            self._coefficients = np.asarray(filters)
            self.Nf = self._coefficients[1]

    def _evaluate(self, x, method='recursive'):

        if x.min() < 0 or x.max() > self.G.lmax:
            _logger.warning('You are trying to evaluate Chebyshev '
                            'polynomials outside of their orthonormal '
                            'domain [0, {:.2f}]'.format(self.G.lmax))

        x = np.asarray(x)
        x = 2 * x / self.G.lmax - 1  # [0, lmax] => [-1, 1]

        return self._evaluate_polynomials(x)

    def _filter(self, s, method='recursive', _=None):
        # method = 'clenshaw' in constructor or filter?
        L = rescale_L(G.L)
        return self._apply_polynomials(L, s)

    def _compute_coefficients(self, filters):
        pass

    def _evaluate_polynomials(self, y, s=1):
        pass
