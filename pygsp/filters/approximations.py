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

    Evaluation methods (which can be passed when calling :meth:`Filter.evaluate` or :meth:`Filter.filter` are:

    * recursive, defined
    * direct, which returns :math:`\sum_k c_k T_k(x) s = \sum_k c_k \cos(k \arccos x) s`.

    Parameters
    ----------
    G : graph
    filters : Filter or array-like
        Either a :class:`Filter` object or a set of Chebyshev coefficients
        represented as an array of size K x F, where K is the polynomial
        order and F the number of filters.
        K x Fout x Fin
        For convenience, Fin and Fout can be omitted.
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
            while self._coefficients.ndim < 3:
                self._coefficients = np.expand_dims(self._coefficients, -1)
            self.Nf = self._coefficients.shape[1]

    def _evaluate(self, x, method):

        if x.min() < 0 or x.max() > self.G.lmax:
            _logger.warning('You are trying to evaluate Chebyshev '
                            'polynomials outside of their orthonormal '
                            'domain [0, G.lmax={:.2f}].'.format(self.G.lmax))

        x = 2 * x / self.G.lmax - 1  # [0, lmax] => [-1, 1]

        # The recursive method is the fastest.
        method = 'recursive' if method is None else method

        return getattr(self, '_evaluate_' + method)(x)

    def _filter(self, s, method, _):
        # method = 'clenshaw' in constructor or filter?
        # Might be faster with signals in fortran-contiguous format.
        # s: N_SIGNALS x N_FEATURES x N_NODES

        L = self.G.L
        if not sparse.issparse(L):
            I = np.identity(self.G.N, dtype=L.dtype)
        else:
            I = sparse.identity(self.G.N, format=L.format, dtype=L.dtype)

        L = 2 * L / self.G.lmax - I  # [0, lmax] => [-1, 1]

        # The recursive method is the fastest.
        method = 'recursive' if method is None else method

        return getattr(self, '_filter_' + method)(L, s)

    def _compute_coefficients(self, filters):
        r"""Compute the coefficients of the Chebyshev series approximating the filters.

        Some implementations define c_0 / 2.
        """
        pass

    def _evaluate_direct(self, x):
        r"""Evaluate Fout*Fin polynomials at each value in x."""
        c = self._coefficients
        K, Fout, Fin = c.shape  # #order x #features_out x #features_in
        c = c.reshape((K, Fout * Fin) + (1,) * x.ndim)  # For broadcasting.
        result = np.zeros((Fout * Fin,) + x.shape)
        x = np.arccos(x)
        for k in range(K):
            result += c[k] * np.cos(k * x)
        return result.reshape((Fout, Fin) + x.shape).squeeze()

    def _evaluate_recursive(self, x):
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

        c = self._coefficients
        K, Fout, Fin = c.shape  # #order x #features_out x #features_in
        c = c.reshape((K, Fout * Fin) + (1,) * x.ndim)  # For broadcasting.

        x0 = np.ones_like(x)
        result = c[0] * x0
        if K > 1:
            x1 = x
            result += c[1] * x1
        for k in range(2, K):
            x2 = 2 * x * x1 - x0
            result += c[k] * x2
            x0, x1 = x1, x2
        return result.reshape((Fout, Fin) + x.shape).squeeze()

    def _filter_recursive(self, L, s):
        r"""Filter a signal with the 3-way recursive relation.
        Time: O(M N Fin Fout K)
        Space: O(4 M Fin N)
        """
        c = self._coefficients
        K, Fout, Fin = c.shape  # #order x #features_out x #features_in
        M, Fin, N = s.shape  # #signals x #features x #nodes

        def mult(c, x):
            """Multiply the diffused signals by the Chebyshev coefficients."""
            x.shape = (N, Fin, M)
            return np.einsum('oi,nim->nom', c, x)
        def dot(L, x):
            """One diffusion step by multiplication with the Laplacian."""
            x.shape = (N, Fin * M)
            return L.dot(x)

        x0 = np.asarray(s.T, order='C')
        result = mult(c[0], x0)
        if K > 1:
            x1 = dot(L, x0)
            result += mult(c[1], x1)
        for k in range(2, K):
            x2 = 2 * dot(L, x1) - x0
            result += mult(c[k], x2)
            x0, x1 = x1, x2
        return result.T
