# -*- coding: utf-8 -*-

import copy

import numpy as np

from pygsp import utils
# prevent circular import in Python < 3.5
from . import approximations
from ..operators.transforms import gft, igft


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
        The graph to which the filterbank is tailored.
    filters : function or list of functions
        A (list of) function defining the filterbank. One function per filter.

    Attributes
    ----------
    G : Graph
        The graph to which the filterbank was tailored. It is a reference to
        the graph passed when instantiating the class.
    g : function or list of functions
        A (list of) function defining the filterbank. One function per filter.
        Either passed by the user when instantiating the base class, either
        constructed by the derived classes.
    Nf : int
        Number of filters in the filterbank.

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, filters
    >>>
    >>> G = graphs.Logo()
    >>> my_filter = filters.Filter(G, lambda x: x / (1. + x))
    >>>
    >>> # Signal: Kronecker delta.
    >>> signal = np.zeros(G.N)
    >>> signal[42] = 1
    >>>
    >>> filtered_signal = my_filter.analysis(signal)

    """

    def __init__(self, G, filters):

        self.G = G

        if isinstance(filters, list):
            self.g = filters
        else:
            self.g = [filters]

        self.Nf = len(self.g)

    def analysis(self, s, method='chebyshev', order=30):
        r"""
        Analyze, or filter, a signal with the filter bank.

        The method computes the transform coefficients of a signal :math:`s`,
        where the atoms of the transform dictionary are generalized
        translations of each graph spectral filter to each vertex on the graph:

        .. math:: c = D^* s,

        where the columns of :math:`D` are :math:`g_{i,m} = T_i g_m` and
        :math:`T_i` is a generalized translation operator applied to each
        filter :math:`\hat{g}_m(\cdot)`. Each column of :math:`c` is the
        response of the signal to one filter.

        In other words, this function is applying the analysis operator
        :math:`D^*` associated with the frame defined by the filter bank to the
        signals.

        Parameters
        ----------
        s : ndarray
            Graph signals to analyze, a matrix of size N x Ns where N is the
            number of nodes and Ns the number of signals.
        method : 'exact', 'chebyshev', 'lanczos'
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. The Lanczos
            approximation is not working yet.
        order : int
            Degree of the Chebyshev polynomials.

        Returns
        -------
        c : ndarray
            Transform coefficients, a matrix of size Nf*N x Ns where Nf is the
            number of filters, N the number of nodes, and Ns the number of
            signals.

        See Also
        --------
        synthesis : adjoint of the analysis operator

        References
        ----------
        See :cite:`hammond2011wavelets` for more details.

        Examples
        --------
        Create a smooth graph signal by low-pass filtering white noise.

        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> G.estimate_lmax()
        >>> s1 = np.random.uniform(size=(G.N, 4))
        >>> s2 = filters.Expwin(G).analysis(s1)
        >>> G.plot_signal(s1[:, 0])
        >>> G.plot_signal(s2[:, 0])

        """

        if method == 'chebyshev':
            cheb_coef = approximations.compute_cheby_coeff(self, m=order)
            c = approximations.cheby_op(self.G, cheb_coef, s)

        elif method == 'lanczos':
            raise NotImplementedError
            # c = approximations.lanczos_op(self, s, order=order)

        elif method == 'exact':
            N = self.G.N  # nb of nodes
            try:
                Ns = np.shape(s)[1]  # nb signals
                c = np.zeros((N * self.Nf, Ns))
                is2d = True
            except IndexError:
                c = np.zeros((N * self.Nf))
                is2d = False

            fie = self.evaluate(self.G.e)

            if self.Nf == 1:
                if is2d:
                    fs = np.tile(fie, (Ns, 1)).T * gft(self.G, s)
                    return igft(self.G, fs)
                else:
                    fs = fie * gft(self.G, s)
                    return igft(self.G, fs)
            else:
                tmpN = np.arange(N, dtype=int)
                for i in range(self.Nf):
                    if is2d:
                        fs = gft(self.G, s)
                        fs *= np.tile(fie[i], (Ns, 1)).T
                        c[tmpN + N * i] = igft(self.G, fs)
                    else:
                        fs = fie[i] * gft(self.G, s)
                        c[tmpN + N * i] = igft(self.G, fs)

        else:
            raise ValueError('Unknown method: {}'.format(method))

        return c

    @utils.filterbank_handler
    def evaluate(self, x, i=0):
        r"""
        Evaluate the response of the filter bank at frequencies x.

        Parameters
        ----------
        x : ndarray
            Graph frequencies at which to evaluate the filter.
        i : int
            Index of the filter to evaluate. Default: evaluate all of them.

        Returns
        -------
        y : ndarray
            Responses of the filters.

        Examples
        --------
        Frequency response of a low-pass filter.

        >>> import matplotlib.pyplot as plt
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> f = filters.Expwin(G)
        >>> G.compute_fourier_basis()
        >>> y = f.evaluate(G.e)
        >>> plt.plot(G.e, y)  # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at ...>]

        """

        return self.g[i](x)

    def inverse(self, c):
        r"""
        Not implemented yet.
        """
        raise NotImplementedError

    def synthesis(self, c, method='chebyshev', order=30):
        r"""
        Synthesize a signal from its filter bank coefficients.

        The method synthesizes a signal :math:`s` from its coefficients
        :math:`c`, where the atoms of the transform dictionary are generalized
        translations of each graph spectral filter to each vertex on the graph:

        .. math:: s = D c,

        where the columns of :math:`D` are :math:`g_{i,m} = T_i g_m` and
        :math:`T_i` is a generalized translation operator applied to each
        filter :math:`\hat{g}_m(\cdot)`.

        In other words, this function is applying the synthesis operator
        :math:`D` associated with the frame defined from the filter bank to the
        coefficients.

        Parameters
        ----------
        c : ndarray
            Transform coefficients, a matrix of size Nf*N x Ns where Nf is the
            number of filters, N the number of nodes, and Ns the number of
            signals.
        method : 'exact', 'chebyshev', 'lanczos'
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. The Lanczos
            approximation is not working yet.
        order : int
            Degree of the Chebyshev approximation.

        Returns
        -------
        s : ndarray
            Synthesized graph signals, a matrix of size N x Ns where N is the
            number of nodes and Ns the number of signals.

        See Also
        --------
        analysis : adjoint of the synthesis operator

        References
        ----------
        See :cite:`hammond2011wavelets` for more details.

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Sensor(30, seed=42)
        >>> G.estimate_lmax()

        Localized smooth signal:

        >>> s1 = np.zeros((G.N, 1))
        >>> s1[13] = 1
        >>> s1 = filters.Heat(G, tau=3).analysis(s1)

        Filter and reconstruct our signal:

        >>> g = filters.MexicanHat(G, Nf=4)
        >>> c = g.analysis(s1)
        >>> s2 = g.synthesis(c)

        Look how well we were able to reconstruct:

        >>> g.plot()
        >>> G.plot_signal(s1[:, 0])
        >>> G.plot_signal(s2[:, 0])
        >>> print('{:.2f}'.format(np.linalg.norm(s1 - s2)))
        0.30

        Perfect reconstruction with Itersine, a tight frame:

        >>> g = filters.Itersine(G)
        >>> c = g.analysis(s1)
        >>> s2 = g.synthesis(c)
        >>> err = np.linalg.norm(s1 - s2)
        >>> print('{:.2f}'.format(np.linalg.norm(s1 - s2)))
        0.00

        """

        N = self.G.N

        if method == 'exact':
            fie = self.evaluate(self.G.e)
            Nv = np.shape(c)[1]
            s = np.zeros((N, Nv))
            tmpN = np.arange(N, dtype=int)

            if self.Nf == 1:
                fc = np.tile(fie, (Nv, 1)).T * gft(self.G, c[tmpN])
                s += igft(np.conjugate(self.G.U), fc)
            else:
                for i in range(self.Nf):
                    fc = gft(self.G, c[N * i + tmpN])
                    fc *= np.tile(fie[:][i], (Nv, 1)).T
                    s += igft(np.conjugate(self.G.U), fc)

        elif method == 'chebyshev':
            cheb_coeffs = approximations.compute_cheby_coeff(self, m=order,
                                                             N=order+1)
            s = np.zeros((N, np.shape(c)[1]))
            tmpN = np.arange(N, dtype=int)

            for i in range(self.Nf):
                s += approximations.cheby_op(self.G,
                                             cheb_coeffs[i], c[i * N + tmpN])

        elif method == 'lanczos':
            raise NotImplementedError
            s = np.zeros((N, np.shape(c)[1]))
            tmpN = np.arange(N, dtype=int)

            for i in range(self.Nf):
                s += approximations.lanczos_op(self.G, self.g[i],
                                               c[i * N + tmpN], order=order)

        else:
            raise ValueError('Unknown method: {}'.format(method))

        return s

    def approx(self, m, N):
        r"""
        Not implemented yet.
        """
        raise NotImplementedError

    def tighten(self):
        r"""
        Not implemented yet.
        """
        raise NotImplementedError

    def estimate_frame_bounds(self, min=0, max=None, N=1000,
                              use_eigenvalues=False):
        r"""
        Compute approximate frame bounds for the filterbank.

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
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> G.estimate_lmax()
        >>> f = filters.MexicanHat(G)

        Bad estimation:

        >>> A, B = f.estimate_frame_bounds(min=1, max=20, N=100)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.126, B=0.270

        Better estimation:

        >>> A, B = f.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.177, B=0.270

        Best estimation:

        >>> G.compute_fourier_basis()
        >>> A, B = f.estimate_frame_bounds(use_eigenvalues=True)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.178, B=0.270

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

        sum_filters = np.sum(np.abs(np.power(self.evaluate(x), 2)), axis=0)

        return sum_filters.min(), sum_filters.max()

    def compute_frame(self, **kwargs):
        r"""
        Compute the frame associated with the filter bank.

        The size of the returned matrix operator :math:`D` is N x MN, where M
        is the number of filters and N the number of nodes. Multiplying this
        matrix with a set of signals is equivalent to analyzing them with the
        associated filterbank.

        Parameters
        ----------
        kwargs: dict
            Parameters to be passed to the :meth:`analysis` method.

        Returns
        -------
        frame : ndarray
            Matrix of size N x MN.

        See also
        --------
        analysis: more efficient way to filter signals

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>>
        >>> G = graphs.Logo()
        >>> G.estimate_lmax()
        >>>
        >>> f = filters.MexicanHat(G)
        >>> frame = f.compute_frame()
        >>> print('{} nodes, matrix is {} x {}'.format(G.N, *frame.shape))
        1130 nodes, matrix is 1130 x 6780
        >>>
        >>> s = np.random.uniform(size=G.N)
        >>> c1 = frame.T.dot(s)
        >>> c2 = f.analysis(s)
        >>>
        >>> np.linalg.norm(c1 - c2) < 1e-10
        True

        """

        N = self.G.N

        if N > 2000:
            _logger.warning('Creating a big matrix, you can use other means.')

        Ft = self.analysis(np.identity(N), **kwargs)
        F = np.empty(Ft.T.shape)
        tmpN = np.arange(N, dtype=int)

        for i in range(self.Nf):
            F[:, N * i + tmpN] = Ft[N * i + tmpN]

        return F

    def can_dual(self):
        r"""
        Creates a dual graph form a given graph
        """
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

        gdual = copy.deepcopy(self)

        for i in range(self.Nf):
            gdual.g[i] = lambda x, ind=i: can_dual_func(self, ind,
                                                        copy.deepcopy(x))

        return gdual

    def plot(self, **kwargs):
        r"""
        Plot the filter.

        See :func:`pygsp.plotting.plot_filter`.
        """
        from pygsp import plotting
        plotting.plot_filter(self, **kwargs)
