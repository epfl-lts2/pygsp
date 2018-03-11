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
            self.Nf = self._coefficients.shape[1]

    def _evaluate(self, x, method='recursive'):

        if x.min() < 0 or x.max() > self.G.lmax:
            _logger.warning('You are trying to evaluate Chebyshev '
                            'polynomials outside of their orthonormal '
                            'domain [0, {:.2f}].'.format(self.G.lmax))

        x = 2 * x / self.G.lmax - 1  # [0, lmax] => [-1, 1]

        return self._evaluate_polynomials(x)

    def _filter(self, s, method='recursive', _=None):
        # method = 'clenshaw' in constructor or filter?

        M, M = L.shape
        I = sparse.identity(M, format='csr', dtype=L.dtype)
        L = 2 * L - self.G.lmax / 2 - I

        return self._apply_polynomials(L, s)

    def _compute_coefficients(self, filters):
        r"""Compute the coefficients of the Chebyshev series approximating the filters.

        Some implementations define c_0 / 2.
        """
        pass

    def _evaluate_polynomials(self, y, s=1):
        """Evaluate a Chebyshev series for y. Optionally, times s.

        .. math: p(y) = \sum_{k=0}^{K} a_k * T_k(y) * s

        Parameters
        ----------
        c: array-like
            set of Chebyshev coefficients. (size K x F where K is the polynomial order, F is the number of filters)
        y: array-like
            vector to be evaluated. (size N x 1)
            vector or matrix
        signal: array-like
            signal (vector) to be multiplied to the result. It allows to avoid the computation of powers of matrices when what we care about is L^k s not L^k.
            vector or matrix (ndarray)

        Returns
        -------
        corresponding Chebyshev Series. (size F x N)

        """

        K = self._coefficients.shape[1]
        c = self._coefficients
        # Reshaping the coefficients to use broadcasting.
        c = c.reshape(c.shape + (1,) * y.ndim)

        y0 = np.ones_like(y)
        result = c[0] * y0.dot(s)
        if K > 1:
            y1 = y
            result += c[1] * y1
        for k in range(2, K):
            y2 = 2 * y.dot(s) * y1 - y0
            result += c[k] * y2
            y0, y1 = y1, y2
        return result
