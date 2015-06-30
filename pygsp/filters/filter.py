# -*- coding: utf-8 -*-

from math import log
import numpy as np
from copy import deepcopy
from pygsp import utils, operators


class Filter(object):
    r"""
    Parent class for all Filters or Filterbanks, contains the shared
    methods for those classes.
    """

    def __init__(self, G, filters=None, **kwargs):

        self.logger = utils.build_logger(__name__)

        if not hasattr(G, 'lmax'):
            self.logger.info('{} : has to compute lmax'.format(
                self.__class__.__name__))
            G.lmax = utils.estimate_lmax(G)

        self.G = G

        if filters:
            if isinstance(filters, list):
                self.g = filters
            else:
                self.g = [filters]

    def analysis(self, G, s, method=None, cheb_order=30, **kwargs):
        r"""
        Operator to analyse a filterbank

        Parameters
        ----------
        G : Graph object
        s : ndarray
            graph signal to analyse
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
        >>> co = MH.analysis(G, x)

        Matlab Authors
        --------------
        David I Shuman, Nathanael Perraudin

        :cite:`hammond2011wavelets`

        """
        if type(G) is list:
            output = []
            for i in range(len(self.g)):
                output.append(self.g[i].analysis(G[i]), s[i], method=method,
                              cheb_order=cheb_order, **kwargs)

            return output

        if not method:
            if hasattr(G, 'U'):
                method = 'exact'
            else:
                method = 'cheby'

        Nf = len(self.g)

        self.logger.info('The analysis method is {}'.format(method))

        if method == 'exact':
            if not hasattr(G, 'e') or not hasattr(G, 'U'):
                self.logger.info('The Fourier matrix is not available. '
                                 'The function will compute it for you.')
                operators.compute_fourier_basis(G)

            try:
                Nv = np.shape(s)[1]
                c = np.zeros((G.N * Nf, Nv))
                is2d = True
            except IndexError:
                c = np.zeros((G.N * Nf))
                is2d = False

            fie = self.evaluate(G.e)

            if Nf == 1:
                if is2d:
                    return operators.igft(G, np.tile(fie, (Nv, 1)).T*operators.gft(G, s))
                else:
                    return operators.igft(G, fie*operators.gft(G, s))
            else:
                for i in range(Nf):
                    if is2d:
                        c[np.arange(G.N) + G.N*i] = operators.igft(G, np.tile(fie[:][i], (Nv, 1)).T*operators.gft(G, s))
                    else:
                        c[np.arange(G.N) + G.N*i] = operators.igft(G, fie[:][i]*operators.gft(G, s))

        elif method == 'cheby':  # Chebyshev approx
            if not hasattr(G, 'lmax'):
                self.logger.info('FILTER_ANALYSIS: The variable lmax is not '
                                 'available. The function will compute '
                                 'it for you.')
                utils.estimate_lmax(G)

            cheb_coef = operators.compute_cheby_coeff(self, G, m=cheb_order)
            c = operators.cheby_op(G, cheb_coef, s)

        elif method == 'lanczos':
            raise NotImplementedError

        else:
            raise ValueError('Unknown method: please select exact, '
                             'cheby or lanczos')

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
        fd = np.zeros(x.size)
        fd = self.g[i](x)
        return fd

    def inverse(self, G, c, **kwargs):
        raise NotImplementedError

    def synthesis(self, G, c, order=30, method=None, **kwargs):
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
        if isinstance(G, list):
            output = []
            for i in range(len(self.g)):
                output.append(self.g[i].synthesis(G[i]), c[i], method=method,
                              order=order, **kwargs)

            return output

        Nf = len(self.g)

        if not method:
            if hasattr(G, 'U'):
                method = 'exact'
            else:
                method = 'cheby'

        if method == 'exact':
            if not hasattr(G, 'e') or not hasattr(G, 'U'):
                self.logger.info("The Fourier matrix is not available. "
                                 "The function will compute it for you.")
                operators.compute_fourier_basis(G)

            fie = self.evaluate(G.e)
            Nv = np.shape(c)[1]
            s = np.zeros((G.N, Nv))

            if Nf == 1:
                s = s + operators.igft(np.conjugate(G.U),
                                       np.tile(fie, (Nv, 1)).T*operators.gft(G, c[G.N*i + np.arange(G.N)]))
            else:
                for i in range(Nf):
                    s = s + operators.igft(np.conjugate(G.U),
                                           np.tile(fie[:][i], (Nv, 1)).T*operators.gft(G, c[G.N*i + np.arange(G.N)]))

            return s

        elif method == 'cheby':
            if hasattr(G, 'lmax'):
                self.logger.info('The variable lmax is not available. '
                                 'The function will compute it for you.')
                utils.estimate_lmax(G)

            cheb_coeffs = operators.compute_cheby_coeff(self, G, m=order,
                                                        N=order+1)
            s = np.zeros((G.N, np.shape(c)[1]))

            for i in range(Nf):
                s = s + operators.cheby_op(G,
                                           cheb_coeffs[i],
                                           c[i*G.N + np.arange(G.N)])

            return s

        elif method == 'lanczos':
            s = np.zeros((G.N, np.shape(c)[1]))

            for i in range(Nf):
                s += utils.lanczos_op(G, self.g[i], c[i*G.N + np.range(G.N)],
                                      order=order)

            return s

        else:
            raise ValueError('Unknown method: please select exact,'
                             ' cheby or lanczos')

    def approx(G, m, N, **kwargs):
        raise NotImplementedError

    def tighten(G):
        raise NotImplementedError

    def filterbank_bounds(self, G, N=999, use_eigenvalues=True):
        r"""
        Compute approximate frame bounds for a filterbank.

        Parameters
        ----------
        G : Graph structure or interval to compute the bound
            (given in an ndarray).
            G = Logo() or G = np.array([xmin, xnmax])
        N : Number of point for the line search
            Default is 999
        use_eigenvalues : Use eigenvalues if possible.
            To be used, the eigenvalues have to be computed first using
            Default is True

        Returns
        -------
        A   : Filterbank lower bound
        B   : Filterbank Upper bound

        """
        if type(G) is list:
            output = []
            for i in range(len(self.g)):
                output.append(self.g[i].analysis(G[i]), N=N,
                              use_eigenvalues=use_eigenvalues)

            return output

        from pygsp.graphs import Graph

        if isinstance(G, Graph):
            if not hasattr(G, 'lmax'):
                utils.estimate_lmax(G)
                self.logger.info('FILTERBANK_BOUNDS: Had to estimate lmax.')
            xmin = 0
            xmax = G.lmax

        else:
            xmin = G[0]
            xmax = G[1]

        if use_eigenvalues and isinstance(G, Graph):
            if hasattr(G, 'e'):
                lamba = G.e

            else:
                raise ValueError('You need to calculate and set the '
                                 'eigenvalues to normalize the kernel: '
                                 'use compute_fourier_basis.')
        else:
            lamba = np.linspace(xmin, xmax, N)

        sum_filters = np.sum(np.abs(np.power(self.evaluate(lamba), 2)), axis=0)

        A = np.min(sum_filters)
        B = np.max(sum_filters)

        return A, B

    def filterbank_matrix(self, G):
        r"""
        Create the matrix of the filterbank frame.

        This function creates the matrix associated to the filterbank g. The\
        size of the matrix is MN x N, where M is the number of filters.

        Parameters
        ----------
        G : Graph

        Returns
        -------
        F : Frame
        """
        if G.N > 2000:
            self.logger.warning('Warning: Create a big matrix, '
                                'you can use other methods.')

        Nf = len(self.g)
        Ft = self.analysis(G, np.identity(G.N))
        F = np.zeros(np.shape(Ft.T))

        for i in range(Nf):
            F[:, G.N*i + np.arange(G.N)] = Ft[G.N*i + np.arange(G.N)]

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
