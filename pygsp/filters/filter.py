# -*- coding: utf-8 -*-

from pygsp import utils
from pygsp.operators import fast_filtering, operator

import numpy as np
from math import log
from copy import deepcopy


class Filter(object):
    r"""Parent class for all Filters or Filterbanks, contains the shared methods for those classes."""

    def __init__(self, G, filters=None, **kwargs):

        self.logger = utils.build_logger(__name__, **kwargs)

        if not hasattr(G, 'lmax'):
            self.logger.info('{} : has to compute lmax'.format(
                self.__class__.__name__))
            G.estimate_lmax()

        self.G = G

        if filters:
            if isinstance(filters, list):
                self.g = filters
            else:
                self.g = [filters]
        else:
            self.g = []

    def analysis(self, s, method=None, cheb_order=30, lanczos_order=30, **kwargs):
        r"""
        Operator to analyse a filterbank

        Parameters
        ----------
        s : ndarray
            graph signals to analyse
        method : string
            wether using an exact method, cheby approx or lanczos
        cheb_order : int
            Order for chebyshev

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

        :cite:`hammond2011wavelets`

        """
        if not method:
            method = 'exact' if hasattr(self.G, 'U') else 'cheby'
        self.logger.info('The analysis method is {}'.format(method))

        if method == 'cheby':  # Chebyshev approx
            if not hasattr(self.G, 'lmax'):
                self.logger.info('FILTER_ANALYSIS: computing lmax.')
                self.G.estimate_lmax()

            cheb_coef = fast_filtering.compute_cheby_coeff(self, m=cheb_order)
            c = fast_filtering.cheby_op(self.G, cheb_coef, s)

        elif method == 'lanczos':  # Lanczos approx
            c = fast_filtering.lanczos_op(self, s, self.G, order=lanczos_order)

        elif method == 'exact':  # Exact computation
            if not hasattr(self.G, 'e') or not hasattr(self.G, 'U'):
                self.logger.info('The Fourier matrix is not available. '
                                 'The function will compute it for you.')
                self.G.compute_fourier_basis()

            Nf = len(self.g)  # nb of filters
            N = self.G.N  # nb of nodes
            try:
                Ns = np.shape(s)[1]  # nb signals
                c = np.zeros((N * Nf, Ns))
                is2d = True
            except IndexError:
                c = np.zeros((N * Nf))
                is2d = False

            fie = self.evaluate(self.G.e)

            if Nf == 1:
                if is2d:
                    return operator.igft(self.G, np.tile(fie, (Ns, 1)).T *
                                         operator.gft(self.G, s))
                else:
                    return operator.igft(self.G, fie * operator.gft(self.G, s))
            else:
                tmpN = np.arange(N, dtype=int)
                for i in range(Nf):
                    if is2d:
                        c[tmpN + N*i] = operator.igft(self.G, np.tile(fie[i], (Ns, 1)).T *
                                                      operator.gft(self.G, s))
                    else:
                        c[tmpN + N*i] = operator.igft(self.G, fie[i] *
                                                      operator.gft(self.G, s))


        else:
            raise ValueError('Unknown method: please select exact, '
                             'cheby or lanczos')

        return c

    @utils.filterbank_handler
    def evaluate(self, x, *args, **kwargs):
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
        i = kwargs.pop('i', 0)

        fd = np.zeros(x.size)
        fd = self.g[i](x)
        return fd

    def inverse(self, c, **kwargs):
        raise NotImplementedError

    def synthesis(self, c, order=30, method=None, **kwargs):
        r"""
        Synthesis operator of a filterbank

        Parameters
        ----------
        G : Graph structure.
        c : Transform coefficients
        method : Select the method ot be used for the computation.
            - 'exact' : Exact method using the graph Fourier matrix
            - 'cheby' : Chebyshev polynomial approximation
            - 'lanczos' : Lanczos approximation

            Default : if the Fourier matrix is present: 'exact' otherwise
            'cheby'
        order : Degree of the Chebyshev approximation
            Default is 30

        Returns
        -------
        signal : sythesis signal

        Examples
        --------

        Reference
        ----------
        See :cite:`hammond2011wavelets` for more details.

        """
        Nf = len(self.g)
        N = self.G.N

        if not method:
            if hasattr(self.G, 'U'):
                method = 'exact'
            else:
                method = 'cheby'

        if method == 'exact':
            if not hasattr(self.G, 'e') or not hasattr(self.G, 'U'):
                self.logger.info("The Fourier matrix is not available. "
                                 "The function will compute it for you.")
                self.G.compute_fourier_basis()

            fie = self.evaluate(self.G.e)
            Nv = np.shape(c)[1]
            s = np.zeros((N, Nv))
            tmpN = np.arange(N, dtype=int)

            if Nf == 1:
                s += operator.igft(np.conjugate(self.G.U),
                                   np.tile(fie, (Nv, 1)).T *
                                   operator.gft(self.G, c[tmpN]))
            else:
                for i in range(Nf):
                    s += operator.igft(np.conjugate(self.G.U),
                                       np.tile(fie[:][i], (Nv, 1)).T *
                                       operator.gft(self.G, c[N*i + tmpN]))

        elif method == 'cheby':
            if hasattr(self.G, 'lmax'):
                self.logger.info('The variable lmax is not available. '
                                 'The function will compute it for you.')
                self.G.estimate_lmax()

            cheb_coeffs = operator.compute_cheby_coeff(self, m=order, N=order + 1)
            s = np.zeros((N, np.shape(c)[1]))
            tmpN = np.arange(N, dtype=int)

            for i in range(Nf):
                s = s + operator.cheby_op(self.G, cheb_coeffs[i], c[i*N + tmpN])

        elif method == 'lanczos':
            s = np.zeros((N, np.shape(c)[1]))
            tmpN = np.arange(N, dtype=int)

            for i in range(Nf):
                s += fast_filtering.lanczos_op(self.G, self.g[i], c[i*N + tmpN],
                                               order=order)

        else:
            raise ValueError('Unknown method: please select exact,'
                             ' cheby or lanczos')

        return s

    def approx(m, N, **kwargs):
        raise NotImplementedError

    def tighten():
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

        """
        if bounds:
            xmin, xmax = bounds
            rng = np.linspace(xmin, xmax, N)

        else:
            if not hasattr(self.G, 'lmax'):
                self.logger.info('FILTERBANK_BOUNDS: Has to estimate lmax.')
                self.G.estimate_lmax()

            if not hasattr(self.G, 'e'):
                self.logger.info('FILTERBANK_BOUNDS: Has to compute Fourier basis.')
                self.G.compute_fourier_basis()

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

        """
        N = self.G.N

        if N > 2000:
            self.logger.warning('Create a big matrix, you can use other methods.')

        Nf = len(self.g)
        Ft = self.analysis(np.identity(N))
        F = np.zeros(np.shape(Ft.T))
        tmpN = np.arange(N, dtype=int)

        for i in range(Nf):
            F[:, N*i + tmpN] = Ft[N*i + tmpN]

        return F

    def wlog_scales(self, lmin, lmax, Nscales, t1=1, t2=2):
        r"""
        Compute logarithm scales for wavelets

        Parameters
        ----------
        lmin : int
            Minimum non-zero eigenvalue
        lmax : int
            Maximum eigenvalue
        Nscales : int
            Number of scales

        Returns
        -------
        s : ndarray
            Scale

        """
        smin = t1/lmax
        smax = t2/lmin

        s = np.exp(np.linspace(log(smax), log(smin), Nscales))

        return s

    def can_dual(self):
        r"""
        Creates a dual graph form a given graph
        """
        def can_dual_func(g, n, x):
            # Nshape = np.shape(x)
            x = np.ravel(x)
            N = np.shape(x)[0]
            M = len(g.g)
            gcoeff = g.evaluate(x).T

            s = np.zeros((N, M))
            for i in range(N):
                    s[i] = np.linalg.pinv(np.expand_dims(gcoeff[i], axis=1))

            ret = s[:, n]
            return ret

        gdual = deepcopy(self)

        Nf = len(self.g)
        for i in range(Nf):
            gdual.g[i] = lambda x, ind=i: can_dual_func(self, ind, deepcopy(x))

        return gdual

    def plot(self, **kwargs):
        r"""
        Plot the filter.

        See plotting doc.

        """
        from pygsp import plotting
        plotting.plot_filter(self, **kwargs)
        plotting.show(getattr(kwargs, 'block', False))
