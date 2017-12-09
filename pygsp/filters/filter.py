# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
# prevent circular import in Python < 3.5
from . import approximations


_logger = utils.build_logger(__name__)


class Filter(object):
    r"""
    The base Filter class.

    * Provide a common interface (and implementation) to filter objects.
    * Can be instantiated to construct custom filters from functions.
    * Initialize attributes for derived classes.

    Parameters
    ----------
    G : graph
        The graph to which the filter bank is tailored.
    kernels : function or list of functions
        A (list of) function(s) defining the filter bank in the Fourier domain.
        One function per filter.

    Attributes
    ----------
    G : Graph
        The graph to which the filter bank was tailored. It is a reference to
        the graph passed when instantiating the class.
    kernels : function or list of functions
        A (list of) function defining the filter bank. One function per filter.
        Either passed by the user when instantiating the base class, either
        constructed by the derived classes.
    Nf : int
        Number of filters in the filter bank.

    Examples
    --------
    >>>
    >>> G = graphs.Logo()
    >>> my_filter = filters.Filter(G, lambda x: x / (1. + x))
    >>>
    >>> # Signal: Kronecker delta.
    >>> signal = np.zeros(G.N)
    >>> signal[42] = 1
    >>>
    >>> filtered_signal = my_filter.filter(signal)

    """

    def __init__(self, G, kernels):

        self.G = G

        try:
            iter(kernels)
        except TypeError:
            kernels = [kernels]
        self._kernels = kernels

        self.Nf = len(kernels)

    def evaluate(self, x):
        r"""Evaluate the kernels at given frequencies.

        Parameters
        ----------
        x : ndarray
            Graph frequencies at which to evaluate the filter.

        Returns
        -------
        y : ndarray
            Frequency response of the filters. Shape ``(G.Nf, len(x))``.

        Examples
        --------
        Frequency response of a low-pass filter:

        >>> import matplotlib.pyplot as plt
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> f = filters.Expwin(G)
        >>> G.compute_fourier_basis()
        >>> y = f.evaluate(G.e)
        >>> plt.plot(G.e, y[0])  # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at ...>]

        """
        # Avoid to copy data as with np.array([g(x) for g in self._kernels]).
        y = np.empty((self.Nf, len(x)))
        for i, g in enumerate(self._kernels):
            y[i] = g(x)
        return y

    def filter(self, s, method='chebyshev', order=30):
        r"""Filter signals (analysis or synthesis).

        A signal is defined as a rank-3 tensor of shape ``(N_NODES, N_SIGNALS,
        N_FEATURES)``, where ``N_NODES`` is the number of nodes in the graph,
        ``N_SIGNALS`` is the number of independent signals, and ``N_FEATURES``
        is the number of features which compose a graph signal, or the
        dimensionality of a graph signal. For example if you filter a signal
        with a filter bank of 8 filters, you're extracting 8 features and
        decomposing your signal into 8 parts. That is called analysis. Your are
        thus transforming your signal tensor from ``(G.N, 1, 1)`` to ``(G.N, 1,
        8)``. Now you may want to combine back the features to form an unique
        signal. For this you apply again 8 filters, one filter per feature, and
        sum the result up. As such you're transforming your ``(G.N, 1, 8)``
        tensor signal back to ``(G.N, 1, 1)``. That is known as synthesis. More
        generally, you may want to map a set of features to another, though
        that is not implemented yet.

        The method computes the transform coefficients of a signal :math:`s`,
        where the atoms of the transform dictionary are generalized
        translations of each graph spectral filter to each vertex on the graph:

        .. math:: c = D^* s,

        where the columns of :math:`D` are :math:`g_{i,m} = T_i g_m` and
        :math:`T_i` is a generalized translation operator applied to each
        filter :math:`\hat{g}_m(\cdot)`. Each column of :math:`c` is the
        response of the signal to one filter.

        In other words, this function is applying the analysis operator
        :math:`D^*`, respectively the synthesis operator :math:`D`, associated
        with the frame defined by the filter bank to the signals.

        Parameters
        ----------
        s : ndarray
            Graph signals, a tensor of shape ``(N_NODES, N_SIGNALS,
            N_FEATURES)``, where ``N_NODES`` is the number of nodes in the
            graph, ``N_SIGNALS`` the number of independent signals you want to
            filter, and ``N_FEATURES`` is either 1 (analysis) or the number of
            filters in the filter bank (synthesis).
        method : {'exact', 'chebyshev'}
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. A Lanczos
            approximation is coming.
        order : int
            Degree of the Chebyshev polynomials.

        Returns
        -------
        s : ndarray
            Graph signals, a tensor of shape ``(N_NODES, N_SIGNALS,
            N_FEATURES)``, where ``N_NODES`` and ``N_SIGNALS`` are the number
            of nodes and signals of the signal tensor that pas passed in, and
            ``N_FEATURES`` is either 1 (synthesis) or the number of filters in
            the filter bank (analysis).

        References
        ----------
        See :cite:`hammond2011wavelets` for details on filtering graph signals.

        Examples
        --------

        Create a bunch of smooth signals by low-pass filtering white noise:

        >>> import matplotlib.pyplot as plt
        >>> G = graphs.Ring(N=60)
        >>> G.estimate_lmax()
        >>> s = np.random.RandomState(42).uniform(size=(G.N, 10))
        >>> taus = [1, 10, 100]
        >>> s = filters.Heat(G, taus).filter(s)
        >>> s.shape
        (60, 10, 3)

        Plot the 3 smoothed versions of the 10th signal:

        >>> fig, ax = plt.subplots()
        >>> G.set_coordinates('line1D')  # To visualize multiple signals in 1D.
        >>> G.plot_signal(s[:, 9, :], ax=ax)
        >>> legend = [r'$\tau={}$'.format(t) for t in taus]
        >>> ax.legend(legend)  # doctest: +ELLIPSIS
        <matplotlib.legend.Legend object at ...>

        Low-pass filter a delta to create a localized smooth signal:

        >>> G = graphs.Sensor(30, seed=42)
        >>> G.compute_fourier_basis()  # Reproducible computation of lmax.
        >>> s1 = np.zeros(G.N)
        >>> s1[13] = 1
        >>> s1 = filters.Heat(G, 3).filter(s1)
        >>> s1.shape
        (30,)

        Filter and reconstruct our signal:

        >>> g = filters.MexicanHat(G, Nf=4)
        >>> s2 = g.analyze(s1)
        >>> s2.shape
        (30, 4)
        >>> s2 = g.synthesize(s2)
        >>> s2.shape
        (30,)

        Look how well we were able to reconstruct:

        >>> fig, axes = plt.subplots(1, 2)
        >>> G.plot_signal(s1, ax=axes[0])
        >>> G.plot_signal(s2, ax=axes[1])
        >>> print('{:.5f}'.format(np.linalg.norm(s1 - s2)))
        0.29620

        Perfect reconstruction with Itersine, a tight frame:

        >>> g = filters.Itersine(G)
        >>> s2 = g.analyze(s1, method='exact')
        >>> s2 = g.synthesize(s2, method='exact')
        >>> np.linalg.norm(s1 - s2) < 1e-10
        True

        """
        if s.shape[0] != self.G.N:
            raise ValueError('First dimension should be the number of nodes '
                             'G.N = {}, got {}.'.format(self.G.N, s.shape))

        # TODO: not in self.Nin (Nf = Nin x Nout).
        if s.ndim == 1 or s.shape[-1] not in [1, self.Nf]:
            if s.ndim == 3:
                raise ValueError('Third dimension (#features) should be '
                                 'either 1 or the number of filters Nf = {}, '
                                 'got {}.'.format(self.Nf, s.shape))
            s = np.expand_dims(s, -1)
        n_features_in = s.shape[-1]

        if s.ndim < 3:
            s = np.expand_dims(s, 1)
        n_signals = s.shape[1]

        if s.ndim > 3:
            raise ValueError('At most 3 dimensions: '
                             '#nodes x #signals x #features.')
        assert s.ndim == 3

        # TODO: generalize to 2D (m --> n) filter banks.
        # Only 1 --> Nf (analysis) and Nf --> 1 (synthesis) for now.
        n_features_out = self.Nf if n_features_in == 1 else 1

        if method == 'exact':

            # TODO: will be handled by g.adjoint().
            axis = 1 if n_features_in == 1 else 2
            f = self.evaluate(self.G.e)
            f = np.expand_dims(f.T, axis)
            assert f.shape == (self.G.N, n_features_in, n_features_out)

            s = self.G.gft(s)
            s = np.matmul(s, f)
            s = self.G.igft(s)

        elif method == 'chebyshev':

            # TODO: update Chebyshev implementation (after 2D filter banks).
            c = approximations.compute_cheby_coeff(self, m=order)

            if n_features_in == 1:  # Analysis.
                s = s.squeeze(axis=2)
                s = approximations.cheby_op(self.G, c, s)
                s = s.reshape((self.G.N, n_features_out, n_signals), order='F')
                s = s.swapaxes(1, 2)

            elif n_features_in == self.Nf:  # Synthesis.
                s = s.swapaxes(1, 2)
                s_in = s.reshape(
                    (self.G.N * n_features_in, n_signals), order='F')
                s = np.zeros((self.G.N, n_signals))
                tmpN = np.arange(self.G.N, dtype=int)
                for i in range(n_features_in):
                    s += approximations.cheby_op(self.G,
                                                 c[i],
                                                 s_in[i * self.G.N + tmpN])
                s = np.expand_dims(s, 2)

        else:
            raise ValueError('Unknown method {}.'.format(method))

        # Return a 1D signal if e.g. a 1D signal was filtered by one filter.
        return s.squeeze()

    def analyze(self, s, method='chebyshev', order=30):
        r"""Convenience alias to :meth:`filter`."""
        if s.ndim == 3 and s.shape[-1] != 1:
            raise ValueError('Last dimension (#features) should be '
                             '1, got {}.'.format(s.shape))
        return self.filter(s, method, order)

    def synthesize(self, s, method='chebyshev', order=30):
        r"""Convenience wrapper around :meth:`filter`.

        Will be an alias to `adjoint().filter()` in the future.
        """
        if s.shape[-1] != self.Nf:
            raise ValueError('Last dimension (#features) should be the number '
                             'of filters Nf = {}, got {}.'.format(self.Nf,
                                                                  s.shape))
        return self.filter(s, method, order)

    def localize(self, i, **kwargs):
        r"""Localize the kernels at a node (to visualize them).

        That is particularly useful to visualize a filter in the vertex domain.

        A kernel is localized by filtering a Kronecker delta, i.e.

        .. math:: g(L) s = g(L)_i, \text{ where } s_j = \delta_{ij} =
                  \begin{cases} 0 \text{ if } i \neq j \\
                                1 \text{ if } i = j    \end{cases}

        Parameters
        ----------
        i : int
            Index of the node where to localize the kernel.
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        s : ndarray
            Kernel localized at vertex i.

        Examples
        --------
        Visualize heat diffusion on a grid by localizing the heat kernel.

        >>> import matplotlib
        >>> N = 20
        >>> DELTA = N//2 * (N+1)
        >>> G = graphs.Grid2d(N)
        >>> G.estimate_lmax()
        >>> g = filters.Heat(G, 100)
        >>> s = g.localize(DELTA)
        >>> G.plot_signal(s, highlight=DELTA)

        """
        s = np.zeros(self.G.N)
        s[i] = 1
        return np.sqrt(self.G.N) * self.filter(s, **kwargs)

    def estimate_frame_bounds(self, min=0, max=None, N=1000,
                              use_eigenvalues=False):
        r"""Estimate lower and upper frame bounds.

        The frame bounds are estimated using the vector :code:`np.linspace(min,
        max, N)` with min=0 and max=G.lmax by default. The eigenvalues G.e can
        be used instead if you set use_eigenvalues to True.

        Parameters
        ----------
        min : float
            The lowest value the filter bank is evaluated at. By default
            filtering is bounded by the eigenvalues of G, i.e. min = 0.
        max : float
            The largest value the filter bank is evaluated at. By default
            filtering is bounded by the eigenvalues of G, i.e. max = G.lmax.
        N : int
            Number of points where the filter bank is evaluated.
            Default is 1000.
        use_eigenvalues : bool
            Set to True to use the Laplacian eigenvalues instead.

        Returns
        -------
        A : float
            Lower frame bound of the filter bank.
        B : float
            Upper frame bound of the filter bank.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> f = filters.MexicanHat(G)

        Bad estimation:

        >>> A, B = f.estimate_frame_bounds(min=1, max=20, N=100)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.125, B=0.270

        Better estimation:

        >>> A, B = f.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.177, B=0.270

        Best estimation:

        >>> G.compute_fourier_basis()
        >>> A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.177, B=0.270

        The Itersine filter bank defines a tight frame:

        >>> f = filters.Itersine(G)
        >>> A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.000, B=1.000

        """
        if max is None:
            max = self.G.lmax

        if use_eigenvalues:
            x = self.G.e
        else:
            x = np.linspace(min, max, N)

        sum_filters = np.sum(np.abs(self.evaluate(x)**2), axis=0)

        return sum_filters.min(), sum_filters.max()

    def compute_frame(self, **kwargs):
        r"""Compute the associated frame.

        The size of the returned matrix operator :math:`D` is N x MN, where M
        is the number of filters and N the number of nodes. Multiplying this
        matrix with a set of signals is equivalent to analyzing them with the
        associated filterbank. Though computing this matrix is a rather
        inefficient way of doing it.

        The frame is defined as follows:

        .. math:: g_i(L) = U g_i(\Lambda) U^*,

        where :math:`g` is the filter kernel, :math:`L` is the graph Laplacian,
        :math:`\Lambda` is a diagonal matrix of the Laplacian's eigenvalues,
        and :math:`U` is the Fourier basis, i.e. its columns are the
        eigenvectors of the Laplacian.

        Parameters
        ----------
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        frame : ndarray
            Matrix of size N x MN.

        See also
        --------
        filter: more efficient way to filter signals

        Examples
        --------
        Filtering signals as a matrix multiplication.

        >>> G = graphs.Sensor(N=1000, seed=42)
        >>> G.estimate_lmax()
        >>> f = filters.MexicanHat(G, Nf=6)
        >>> s = np.random.uniform(size=G.N)
        >>>
        >>> frame = f.compute_frame()
        >>> frame.shape
        (1000, 1000, 6)
        >>> frame = frame.reshape(G.N, -1).T
        >>> s1 = np.dot(frame, s)
        >>> s1 = s1.reshape(G.N, -1)
        >>>
        >>> s2 = f.filter(s)
        >>> np.all((s1 - s2) < 1e-10)
        True

        """
        if self.G.N > 2000:
            _logger.warning('Creating a big matrix, you can use other means.')

        # Filter one delta per vertex.
        s = np.identity(self.G.N)
        return self.filter(s, **kwargs)

    def can_dual(self):
        r"""Creates a dual graph form a given graph"""

        def can_dual_func(g, n, x):
            # Nshape = np.shape(x)
            x = np.ravel(x)
            N = np.shape(x)[0]
            M = g.Nf
            gcoeff = g.evaluate(x).T

            s = np.zeros((N, M))
            for i in range(N):
                s[i] = np.linalg.pinv(np.expand_dims(gcoeff[i], axis=1))

            ret = s[:, n]
            return ret

        kernels = []
        for i in range(self.Nf):
            kernels.append(lambda x, i=i: can_dual_func(self, i, x))

        return Filter(self.G, kernels)

    def plot(self, **kwargs):
        r"""Plot the filter bank's frequency response.

        See :func:`pygsp.plotting.plot_filter`.
        """
        from pygsp import plotting
        plotting.plot_filter(self, **kwargs)
