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

    Examples
    --------

    Plot the basis formed by the first K Chebyshev polynomials:

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2)
    >>>
    >>> G = graphs.Ring(N=20)
    >>> G.compute_fourier_basis()  # To be exactly orthogonal in vertex domain.
    >>> G.set_coordinates('line1D')
    >>>
    >>> K = 5  # Polynomials of order up to K.
    >>>
    >>> coefficients = np.identity(K)
    >>> f = filters.Chebyshev(G, coefficients)
    >>> s = f.localize(G.N // 2)
    >>> f.plot(sum=False, eigenvalues=False, ax=axes[0])
    >>> G.plot_signal(s.T, ax=axes[1])
    >>>
    >>> _ = axes[0].set_title('Chebysev polynomials in the spectral domain')
    >>> _ = axes[1].set_title('Chebysev polynomials in the ring graph domain')
    >>> _ = axes[0].legend(['order {}'.format(order) for order in range(K)])
    >>> _ = axes[1].legend(['order {}'.format(order) for order in range(K)])

    They are orthogonal in the vertex domain:

    >>> s = s.T.reshape((G.N, -1))
    >>> print(s.T.dot(s))
    [[20.  0.  0.  0.  0.]
     [ 0. 10.  0.  0.  0.]
     [ 0.  0. 10.  0.  0.]
     [ 0.  0.  0. 10.  0.]
     [ 0.  0.  0.  0. 10.]]

    """

    def __init__(self, G, coefficients):

        self.G = G

        coefficients = np.asarray(coefficients)
        while coefficients.ndim < 3:
            coefficients = np.expand_dims(coefficients, -1)

        self.n_features_out, self.n_features_in = coefficients.shape[1:]
        self.shape = (self.n_features_in, self.n_features_out)  # TODO: useful?
        self.n_filters = self.n_features_in * self.n_features_out
        self.Nf = self.n_filters  # TODO: kept for backward compatibility only.
        self._coefficients = coefficients

    # That is a factory method.
    @classmethod
    def from_filter(cls, filters, order=30, n=None):
        r"""Compute the Chebyshev coefficients which approximate the filters.

        The :math:`K+1` coefficients, where :math:`K` is the polynomial order,
        to approximate the function :math:`f` are computed by the discrete
        orthogonality condition as

        .. math:: a_k \approx \frac{2-\delta_{0k}}{N}
                              \sum_{n=0}^{N-1} T_k(x_n) f(x_n),

        where :math:`\delta_{ij}` is the Kronecker delta function and the
        :math:`x_n` are the N shifted Gauss–Chebyshev zeros of :math:`T_N(x)`,
        given by

        .. math:: x_n = \frac{\lambda_\text{max}}{2}
                        \cos\left( \frac{\pi (2k+1)}{2N} \right)
                        + \frac{\lambda_\text{max}}{2}.

        For any N, these approximate coefficients provide an exact
        approximation to the function at :math:`x_k` with a controlled error
        between those points. The exact coefficients are obtained with
        :math:`N=\infty`, thus representing the function exactly at all points
        in :math:`[0, \lambda_\text{max}]`. The rate of convergence depends on
        the function and its smoothness.

        Parameters
        ----------
        filters : filters.Filter
            A filterbank (:class:`Filter`) to be approximated by a set of
            Chebyshev polynomials.
        order : int
            The order of the Chebyshev polynomials.
        n : int
            The number of Gauss–Chebyshev zeros used to approximate the
            coefficients. Defaults to the polynomial order plus one.

        Examples
        --------

        Chebyshev coefficients which approximate a linear function:

        >>> G = graphs.Ring()
        >>> G.estimate_lmax()
        >>> g = filters.Filter(G, lambda x: 2*x)
        >>> h = filters.Chebyshev.from_filter(g, order=4)
        >>> print(', '.join([str(int(c)) for c in h._coefficients]))
        4, 4, 0, 0, 0

        Coefficients of smooth filters decrease rapidly:

        >>> import matplotlib.pyplot as plt
        >>> taus = [5, 10, 20, 50]
        >>> g = filters.Heat(G, tau=taus)
        >>> h = filters.Chebyshev.from_filter(g, order=10)
        >>> fig, axes = plt.subplots(1, 2)
        >>> g.plot(sum=False, ax=axes[0])
        >>> _ = axes[1].plot(h._coefficients.squeeze())
        >>> _ = axes[0].legend(['tau = {}'.format(tau) for tau in taus])
        >>> _ = axes[1].legend(['tau = {}'.format(tau) for tau in taus])

        """
        lmax = filters.G.lmax

        if n is None:
            n = order + 1

        points = np.pi * (np.arange(n) + 0.5) / n

        # The Gauss–Chebyshev zeros of Tk(x), scaled to [0, lmax].
        zeros = lmax/2 * np.cos(points) + lmax/2

        # TODO: compute with scipy.fftpack.dct().
        c = np.empty((order + 1, filters.Nf))
        for i, kernel in enumerate(filters._kernels):
            for k in range(order + 1):
                T_k = np.cos(k * points)  # Chebyshev polynomials of order k.
                c[k, i] = 2 / n * kernel(zeros).dot(T_k)
        c[0, :] /= 2

        return cls(filters.G, c)

    @staticmethod
    def scale_data(x, lmax):
        r"""Given numbers in [0, lmax], scale them to [-1, 1]."""

        if x.min() < 0 or x.max() > lmax:
            _logger.warning('You are trying to evaluate Chebyshev '
                            'polynomials outside of their orthonormal '
                            'domain [0, lmax={:.2f}].'.format(lmax))

        return 2 * x / lmax - 1

    @staticmethod
    def scale_operator(L, lmax):
        r"""Scale an operator's eigenvalues from [0, lmax] to [-1, 1]."""
        if not sparse.issparse(L):
            I = np.identity(L.shape[0], dtype=L.dtype)
        else:
            I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)

        return 2 * L / lmax - I

    def _evaluate(self, x, method):

        x = self.scale_data(x, self.G.lmax)

        c = self._coefficients
        K, Fout, Fin = c.shape  # #order x #features_out x #features_in
        c = c.reshape((K, Fout * Fin) + (1,) * x.ndim)  # For broadcasting.

        # Recursive faster than direct faster than clenshaw.
        method = 'recursive' if method is None else method

        try:
            y = getattr(self, '_evaluate_' + method)(c, x)
        except AttributeError:
            raise ValueError('Unknown method {}.'.format(method))

        return y.reshape((Fout, Fin) + x.shape).squeeze()

    def _filter(self, s, method, _):

        # TODO: signal normalization will move to Filter.filter()

        # Dimension 3: number of nodes.
        if s.shape[-1] != self.G.N:
            raise ValueError('The last dimension should be {}, '
                             'the number of nodes. '
                             'Got instead a signal of shape '
                             '{}.'.format(self.G.N, s.shape))

        # Dimension 2: number of input features.
        if s.ndim == 1:
            s = np.expand_dims(s, 0)
        if s.shape[-2] != self.n_features_in:
            if self.n_features_in == 1 and s.ndim == 2:
                # Dimension can be omitted if there's 1 input feature.
                s = np.expand_dims(s, -2)
            else:
                raise ValueError('The second to last dimension should be {}, '
                                 'the number of input features. '
                                 'Got instead a signal of shape '
                                 '{}.'.format(self.n_features_in, s.shape))

        # Dimension 1: number of independent signals.
        if s.ndim < 3:
            s = np.expand_dims(s, 0)

        if s.ndim > 3:
            raise ValueError('Signals should have at most 3 dimensions: '
                             '#signals x #features x #nodes.')

        assert s.ndim == 3
        assert s.shape[2] == self.G.N  # Number of nodes.
        assert s.shape[1] == self.n_features_in  # Number of input features.
        # n_signals = s.shape[0]

        L = self.scale_operator(self.G.L, self.G.lmax)

        # Recursive and clenshaw are similarly fast.
        method = 'recursive' if method is None else method

        try:
            return getattr(self, '_filter_' + method)(L, s)
        except AttributeError:
            raise ValueError('Unknown method {}.'.format(method))

    def _evaluate_direct(self, c, x):
        r"""Evaluate Fout*Fin polynomials at each value in x."""
        K, F = c.shape[:2]
        result = np.zeros((F,) + x.shape)
        x = np.arccos(x)
        for k in range(K):
            result += c[k] * np.cos(k * x)
        return result

    def _evaluate_recursive(self, c, x):
        """Evaluate a Chebyshev series for y. Optionally, times s.

        .. math:: p(x) = \sum_{k=0}^{K} a_k * T_k(x) * s

        Parameters
        ----------
        c: array-like
            set of Chebyshev coefficients. (size K x F where K is the polynomial order, F is the number of filters)
        x: array-like
            vector to be evaluated. (size N x 1)
            vector or matrix
        signal: array-like
            signal (vector) to be multiplied to the result. It allows to avoid the computation of powers of matrices when what we care about is L^k s not L^k.
            vector or matrix (ndarray)

        Returns
        -------
        corresponding Chebyshev Series. (size F x N)

        """
        K = c.shape[0]
        x0 = np.ones_like(x)
        result = c[0] * x0
        if K > 1:
            x1 = x
            result += c[1] * x1
        for k in range(2, K):
            x2 = 2 * x * x1 - x0
            result += c[k] * x2
            x0, x1 = x1, x2
        return result

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
            x.shape = (M, Fin, N)
            return np.einsum('oi,min->mon', c, x)
        def dot(L, x):
            """One diffusion step by multiplication with the Laplacian."""
            x.shape = (M * Fin, N)
            return L.__rmatmul__(x)  # x @ L

        x0 = s.view()
        result = mult(c[0], x0)
        if K > 1:
            x1 = dot(L, x0)
            result += mult(c[1], x1)
        for k in range(2, K):
            x2 = 2 * dot(L, x1) - x0
            result += mult(c[k], x2)
            x0, x1 = x1, x2
        return result

    def _filter_clenshaw(self, L, s):
        c = self._coefficients
        K, Fout, Fin = c.shape  # #order x #features_out x #features_in
        M, Fin, N = s.shape  # #signals x #features x #nodes

        def mult(c, s):
            """Multiply the signals by the Chebyshev coefficients."""
            return np.einsum('oi,min->mon', c, s)
        def dot(L, x):
            """One diffusion step by multiplication with the Laplacian."""
            x.shape = (M * Fout, N)
            y = L.__rmatmul__(x)  # x @ L
            x.shape = (M, Fout, N)
            y.shape = (M, Fout, N)
            return y

        b2 = 0
        b1 = mult(c[K-1], s) if K >= 2 else np.zeros((M, Fout, N))
        for k in range(K-2, 0, -1):
            b = mult(c[k], s) + 2 * dot(L, b1) - b2
            b2, b1 = b1, b
        return mult(c[0], s) + dot(L, b1) - b2

    def _evaluate_clenshaw(self, c, x):
        K = c.shape[0]
        b2 = 0
        b1 = c[K-1] * np.ones_like(x) if K >= 2 else 0
        for k in range(K-2, 0, -1):
            b = c[k] + 2 * x * b1 - b2
            b2, b1 = b1, b
        return c[0] + x * b1 - b2
