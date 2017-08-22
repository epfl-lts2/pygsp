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
        Operator to analyse a filterbank

        Parameters
        ----------
        s : ndarray
            Graph signals to analyse.
        method : 'exact', 'chebyshev', 'lanczos'
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. The Lanczos
            approximation is not working yet.
        order : int
            Degree of the Chebyshev polynomials.

        Returns
        -------
        c : ndarray
            Transform coefficients

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> MH = filters.MexicanHat(G)
        >>> x = np.arange(G.N**2).reshape(G.N, G.N)
        >>> co = MH.analysis(x)

        References
        ----------
        See :cite:`hammond2011wavelets`

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
        Evaluation of the Filterbank

        Parameters
        ----------
        x = ndarray
            Data
        i = int
            Indice of the filter to evaluate

        Returns
        -------
        fd = ndarray
            Response of the filter

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> MH = filters.MexicanHat(G)
        >>> x = np.arange(2)
        >>> eva = MH.evaluate(x)

        """
        return self.g[i](x)

    def inverse(self, c):
        r"""
        Not implemented yet.
        """
        raise NotImplementedError

    def synthesis(self, c, method='chebyshev', order=30):
        r"""
        Synthesis operator of a filterbank

        Parameters
        ----------
        c : ndarray
            Transform coefficients.
        method : 'exact', 'chebyshev', 'lanczos'
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. The Lanczos
            approximation is not working yet.
        order : int
            Degree of the Chebyshev approximation.

        Returns
        -------
        signal : synthesis signal

        References
        ----------
        See :cite:`hammond2011wavelets` for more details.

        Examples
        --------
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> Nf = 6
        >>>
        >>> vertex_delta = 83
        >>> S = np.zeros((G.N * Nf, Nf))
        >>> S[vertex_delta] = 1
        >>> for i in range(Nf):
        ...     S[vertex_delta + i * G.N, i] = 1
        >>>
        >>> Wk = filters.MexicanHat(G, Nf)
        >>> Sf = Wk.synthesis(S)

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

    def filterbank_bounds(self, N=999, bounds=None):
        r"""
        Compute approximate frame bounds for a filterbank.

        Parameters
        ----------
        bounds : interval to compute the bound.
            Given in an ndarray: np.array([xmin, xnmax]).
            By default, bounds is None and filtering is bounded
            by the eigenvalues of G.
        N : Number of point for the line search
            Default is 999

        Returns
        -------
        lower : Filterbank lower bound
        upper : Filterbank upper bound

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> MH = filters.MexicanHat(G)
        >>> bounds = MH.filterbank_bounds()
        >>> print('lower={:.3f}, upper={:.3f}'.format(bounds[0], bounds[1]))
        lower=0.178, upper=0.270

        """
        if bounds:
            xmin, xmax = bounds
            rng = np.linspace(xmin, xmax, N)
        else:
            rng = self.G.e

        sum_filters = np.sum(np.abs(np.power(self.evaluate(rng), 2)), axis=0)

        return np.min(sum_filters), np.max(sum_filters)

    def filterbank_matrix(self):
        r"""
        Create the matrix of the filterbank frame.

        This function creates the matrix associated to the filterbank g.
        The size of the matrix is MN x N, where M is the number of filters.

        Returns
        -------
        F : Frame

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, filters
        >>> G = graphs.Logo()
        >>> MH = filters.MexicanHat(G)
        >>> matrix = MH.filterbank_matrix()

        """
        N = self.G.N

        if N > 2000:
            _logger.warning('Creating a big matrix, you can use other means.')

        Ft = self.analysis(np.identity(N))
        F = np.zeros(np.shape(Ft.T))
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
