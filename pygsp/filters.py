# -*- coding: utf-8 -*-
r"""
This module implements the main filter class and all the filters subclasses

* :class: `Filter` Main filter class
"""

from math import exp, log, pi
import numpy as np
from numpy import linalg
from copy import deepcopy
import scipy as sp
import scipy.optimize
import pygsp
from pygsp import utils, operators


class Filter(object):
    r"""
    Parent class for all Filters or Filterbanks, contains the shared
    methods for those classes.
    """

    def __init__(self, G, verbose=True, **kwargs):
        self.verbose = verbose
        if not hasattr(G, 'lmax'):
            if self.verbose:
                print(type(self), ': has to compute lmax')
            G.lmax = utils.estimate_lmax(G)
        self.G = G

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
        >>> x = np.arange(64).reshape(8, 8)
        >>> co = MH.analysis(G, x)

        Matlab Authors
        --------------
        David I Shuman, Nathanael Perraudin

        :cite:`hammond2011wavelets`

        """
        if type(G) is list:
            output = []
            for i in range(len(self.g)):
                output.append(g[i].analysis(G[i]), s[i], method=method, cheb_order=cheb_order, **kwargs)

            return output

        if not method:
            if hasattr(G, 'U'):
                method = 'exact'
            else:
                method = 'cheby'

        Nf = len(self.g)

        if method == 'exact':
            if not hasattr(G, 'e') or not hasattr(G, 'U'):
                if self.verbose:
                    print('The Fourier matrix is not available. The function will compute it for you.')
                operators.compute_fourier_basis(G)

            try:
                Nv = np.shape(s)[1]
                c = np.zeros((G.N * Nf, Nv))
            except IndexError:
                Nv = 1
                c = np.zeros((G.N * Nf))

            fie = self.evaluate(G.e)

            if Nf == 1:
                c = operators.igft(G, np.kron(np.ones((Nv)), fie)*operators.gft(G, s))
            else:
                for i in range(Nf):
                    c[np.arange(G.N) + G.N*i] = operators.igft(G, np.kron(np.ones((1, Nv)), np.expand_dims(fie[:][i], axis=1))*operators.gft(G, s))

        elif method == 'cheby':  # Chebyshev approx
            if not hasattr(G, 'lmax'):
                if self.verbose:
                    print('FILTER_ANALYSIS: The variable lmax is not available. The function will compute it for you.')
                utils.estimate_lmax(G)

            cheb_coef = operators.compute_cheby_coeff(self, G, m=cheb_order)
            c = operators.cheby_op(G, cheb_coef, s)

        elif method == 'lanczos':
            raise NotImplementedError

        else:
            raise ValueError('Unknown method: please select exact, cheby or lanczos')

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
        >>> MH.evaluate(x)
        [array([  4.41455329e-01,   6.98096605e-42]),
         array([ 0.        ,  0.20636635]),
         array([ 0.        ,  0.36786227]),
         array([ 0.        ,  0.26561591]),
         array([ 0.        ,  0.13389365]),
         array([ 0.        ,  0.05850726])]

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
            Default : if the Fourier matrix is present: 'exact' otherwise 'cheby'
        order : Degree of the Chebyshev approximation
            Default is 30
        verbose : Verbosity level (False no log - True display warnings)
            Default is True

        Returns
        -------
        signal : sythesis signal

        Examples
        --------

        Reference
        ----------
        See :cite:`hammond2011wavelets` for more details.
        """
        if type(G) is list:
            output = []
            for i in range(len(self.g)):
                output.append(g[i].synthesis(G[i]), c[i], method=method, order=order, **kwargs)

            return output

        Nf = len(self.g)

        if not method:
            if hasattr(G, 'U'):
                method = 'exact'
            else:
                method = 'cheby'

        if method == 'exact':
            if not hasattr(G, 'e') or not hasattr(G, 'U'):
                if self.verbose:
                    print("The Fourier matrix is not available. The function will compute it for you.")
                operators.compute_fourier_basis(G)

            fie = self.evaluate(G.e)
            Nv = np.shape(c)[1]
            s = np.zeros((G.N, Nv))

            for i in range(Nf):
                print(operators.igft(np.conjugate(G.U), np.kron(np.ones((1, Nv)), np.expand_dims(fie[:][i], axis=1))*operators.gft(G, c[G.N*i + np.arange(G.N)])).shape)
                s = s + operators.igft(np.conjugate(G.U), np.kron(np.ones((1, Nv)), np.expand_dims(fie[:][i], axis=1))*operators.gft(G, c[G.N*i + np.arange(G.N)]))

            return s

        elif method == 'cheby':
            if hasattr(G, 'lmax'):
                if self.verbose:
                    print('The variable lmax is not available. The function will compute it for you.')
                utils.estimate_lmax(G)

            cheb_coeffs = operators.compute_cheby_coeff(self, G, m=order, N=order+1)
            s = np.zeros((G.N, np.shape(c)[1]))

            for i in range(Nf):
                s += operators.cheby_op(G, cheb_coeffs[:, i], c[i*G.N + range(G.N)])

            return s

        elif method == 'lanczos':
            s = np.zeros((G.N, np.shape(c)[1]))

            for i in range(Nf):
                s += utils.lanczos_op(G, self[i], c[i*G.N + range(G.N)], order=order, verbose=self.verbose)

            return s

        else:
            raise ValueError('Unknown method: please select exact, cheby or lanczos')

    def approx(G, m, N, **kwargs):
        raise NotImplementedError

    def tighten(G):
        raise NotImplementedError

    def filterbank_bounds(self, G, N=999, use_eigenvalues=True):
        r"""
        Compute approximate frame bounds for a filterbank.

        Parameters
        ----------
        G : Graph structure or interval to compute the bound (given in an ndarray).
            G = Logo() or G = np.array([xmin, xnmax])
        N : Number of point for the line search
            Default is 999
        use_eigenvalues : Use eigenvalues if possible . To be used, the eigenvalues have to be computed first using
            Default is True

        Returns
        -------
        A   : Filterbank lower bound
        B   : Filterbank Upper bound
        """
        if type(G) is list:
            output = []
            for i in range(len(self.g)):
                output.append(g[i].analysis(G[i]), N=N, use_eigenvalues=use_eigenvalues)

            return output

        if isinstance(G, pygsp.graphs.Graph):
            if not hasattr(G, 'lmax'):
                estimate_lmax(G)
                print('FILTERBANK_BOUNDS: Had to estimate lmax.')
            xmin = 0
            xmax = G.lmax

        else:
            xmin = G[0]
            xmax = G[1]

        if use_eigenvalues and isinstance(G, pygsp.graphs.Graph):
            if hasattr(G, 'e'):
                lamba = G.e

            else:
                raise ValueError('You need to calculate and set the eigenvalues to normalize the kernel: use compute_fourier_basis.')
        else:
            lamba = np.linspace(xmin, xmax, N)

        Nf = len(self.g)
        sum_filters = np.sum(np.abs(np.power(self.evaluate(lamba), 2)), axis=0)

        A = np.min(sum_filters)
        B = np.max(sum_filters)

        return A, B

    def filterbank_matrix(self, G):
        r"""
        Create the matrix of the filterbank frame.

        This function create the matrix associated to the filterbank g. The
        size of the matrix is MN x N, where M is the number of filters.

        Parameters
        ----------
        G : Graph
        verbose (bool) : False no log, True print all steps.
            Default is True

        Returns
        -------
        F : Frame
        """
        if self.verbose and G.N > 2000:
            print('Waring. Create a big matrix, you can use other methods.')

        Nf = len(self.g)
        Ft = self.analysis(G, np.identity(G.N))
        F = np.zeros(np.shape(Ft.transpose()))

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
        def can_dual_func(g, n, x):
            Nshape = np.shape(x)
            x = np.ravel(x)
            N = np.shape(x)[0]
            M = len(g.g)
            gcoeff = np.transpose(g.evaluate(x))

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

    def vec2mat(d, Nf):
        raise NotImplementedError

    def mat2vec(d):
        raise NotImplementedError


class Abspline(Filter):
    r"""
    Abspline Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    t : ndarray
        Vector of scale to be used (Initialized by default at
        the value of the log scale)

    Returns
    -------
    out : Abspline

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, **kwargs):
        super(Abspline, self).__init__(G, **kwargs)

        def kernel_abspline3(x, alpha, beta, t1, t2):
            M = np.array([[1, t1, t1**2, t1**3],
                          [1, t2, t2**2, t2**3],
                          [0, 1, 2*t1, 3*t1**2],
                          [0, 1, 2*t2, 3*t2**2]])
            v = np.array([1, 1, t1**(-alpha) * alpha * t1**(alpha-1),
                          -beta*t2**(-(beta+1)) * t2**beta])
            a = np.linalg.solve(M, v)

            r1 = x <= t1
            r2 = (x >= t1)*(x < t2)
            r3 = (x >= t2)

            if isinstance(x, np.float64):

                if r1:
                    r = x[r1]**alpha * t1**(-alpha)
                if r2:
                    r = a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3
                if r3:
                    r = x[r3]**(-beta) * t2**beta

            else:
                r = np.zeros(x.shape)

                x2 = x[r2]

                r[r1] = x[r1]**alpha * t1**(-alpha)
                r[r2] = a[0] + a[1] * x2 + a[2] * x2**2 + a[3] * x2**3
                r[r3] = x[r3]**(-beta) * t2 **beta

            return r

        G.lmin = G.lmax / lpfactor

        if t is None:
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: kernel_abspline3(x, 2, 2, 1, 2)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        self.g = [lambda x: 1.2 * exp(-1) * gl(x / lminfac)]
        for i in range(0, Nf-1):
            self.g.append(lambda x, ind=i: gb(self.t[ind] * x))

        f = lambda x: -gb(x)
        xstar = scipy.optimize.minimize_scalar(f, bounds=(1, 2), method='bounded')
        gamma_l = -f(xstar.x)
        lminfac = .6 * G.lmin
        self.g[0] = lambda x: gamma_l * gl(x / lminfac)


class Expwin(Filter):
    r"""
    Expwin Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    bmax : float
        Maximum relative band (default = 0.2)
    a : int
        Slope parameter (default = 1)

    Returns
    -------
    out : Expwin
    """
    def __init__(self, G, bmax=0.2, a=1., **kwargs):
        super(Expwin, self).__init__(G, **kwargs)

        def fx(x, a):
            y = np.exp(-float(a)/x)
            if isinstance(x, np.ndarray):
                y = np.where(x < 0, 0., y)
            else:
                if x < 0:
                    y = 0.
            return y

        def gx(x, a):
            y = fx(x, a)
            return y/(y + fx(1 - x, a))

        ffin = lambda x, a: gx(1 - x, a)

        g = [lambda x: ffin(np.float64(x)/bmax/G.lmax, a)]
        self.g = g


class HalfCosine(Filter):
    r"""
    HalfCosine Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    Returns
    -------
    out : HalfCosine

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(HalfCosine, self).__init__(G, **kwargs)

        if Nf <= 2:
            raise ValueError('The number of filters must be higher than 2.')

        dila_fact = G.lmax * (3./(Nf - 2))

        main_window = lambda x: np.multiply(np.multiply((.5 + .5*np.cos(2.*pi*(x/dila_fact - 1./2))),
                                                        (x >= 0)),
                                            (x <= dila_fact))

        g = []

        for i in range(Nf):
            g.append(lambda x, ind=i: main_window(x - dila_fact/3. * (ind-2)))

        self.g = g


class Itersine(Filter):
    r"""
    Create a itersine filterbanks

    This function create a itersine half overlap filterbank of Nf filters
    Going from 0 to lambda_max

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax. (default = 6)
    overlap : int
        (default = 2)

    Returns
    -------
    out : Itersine

    """
    def __init__(self, G, Nf=6, overlap=2., **kwargs):
        super(Itersine, self).__init__(G, **kwargs)

        k = lambda x: np.sin(0.5*pi*np.power(np.cos(x*pi), 2)) * ((x >= -0.5)*(x <= 0.5))
        scale = G.lmax/(Nf - overlap + 1.)*overlap
        g = []

        for i in range(1, Nf + 1):
            g.append(lambda x, ind=i: k(x/scale - (ind - overlap/2.)/overlap) / np.sqrt(overlap)*np.sqrt(2))

        self.g = g


class MexicanHat(Filter):
    r"""
    Mexican hat Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    t : ndarray
        Vector of scale to be used (Initialized by default at the value of the log scale)
    normalize : bool
        Wether to normalize the wavelet by the factor/sqrt(t). (default = False)

    Returns
    -------
    out : MexicanHat

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, normalize=False,
                 **kwargs):
        super(MexicanHat, self).__init__(G, **kwargs)

        if t is None:
            G.lmin = G.lmax / lpfactor
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: x * np.exp(-x)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        g = [lambda x: 1.2 * exp(-1) * gl(x / lminfac)]

        for i in range(Nf-1):
            if normalize:
                g.append(lambda x, ind=i: np.sqrt(t[ind]) * gb(self.t[ind] * x))
            else:
                g.append(lambda x, ind=i: gb(self.t[ind] * x))

        self.g = g


class Meyer(Filter):
    r"""
    Meyer Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)

    Returns
    -------
    out : Meyer

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(Meyer, self).__init__(G, **kwargs)

        if not hasattr(G, 't'):
            G.t = (4./(3 * G.lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if self.verbose:
            if len(G.t) >= Nf-1:
                print('You have specified more scales than  the number of scales minus 1')

        t = G.t

        g = [lambda x: kernel_meyer(t[0] * x, 'sf')]
        for i in range(Nf-1):
            g.append(lambda x, ind=i: kernel_meyer(t[ind] * x, 'wavelet'))

        self.g = g

        def kernel_meyer(x, kerneltype):
            r"""
            Evaluates Meyer function and scaling function

            Parameters
            ----------
            x : ndarray
                Array of independant variables values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns
            -------
            r : ndarray

            """

            x = np.array(x)

            l1 = 2/3.
            l2 = 4/3.
            l3 = 8/3.

            v = lambda x: x ** 4. * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1)*(x < l2)
            r3ind = (x >= l2)*(x < l3)

            r = np.empty(x.shape)
            if kerneltype is 'sf':
                r[r1ind] = 1
                r[r2ind] = np.cos((pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kerneltype is 'wavelet':
                r[r2ind] = np.sin((pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise TypeError('Unknown kernel type ', kerneltype)

            return r


class SimpleTf(Filter):
    r"""
    SimpleTf Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    t : ndarray
        Vector of scale to be used (Initialized by default at the value of the log scale)

    Returns
    -------
    out : SimpleTf

    """

    def __init__(self, G, Nf=6, t=None, **kwargs):
        super(SimpleTf, self).__init__(G, **kwargs)

        def kernel_simple_tf(x, kerneltype):
            r"""
            Evaluates 'simple' tight-frame kernel

            Parameters
            ----------
            x : ndarray
                Array if independant variable values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns:
            r : ndarray

            """

            l1 = 0.25
            l2 = 0.5
            l3 = 1.

            h = lambda x: np.sin(pi*x/2.)**2

            r1ind = x < l1
            r2ind = (x >= l1)*(x < l2)
            r3ind = (x >= l2)*(x < l3)

            r = np.zeros(x.shape)
            if kerneltype is 'sf':
                r[r1ind] = 1.
                r[r2ind] = np.sqrt(1 - h(4*x[r2ind] - 1)**2)
            elif kerneltype is 'wavelet':
                r[r2ind] = h(4*(x[r2ind] - 1/4.))
                r[r3ind] = np.sqrt(1 - h(2*x[r3ind] - 1)**2)
            else:
                raise TypeError('Unknown kernel type', kerneltype)

            return r

        if not t:
            t = (1./(2.*G.lmax) * np.power(2, np.arange(Nf-2, -1, -1)))

        if self.verbose:
            if len(t) != Nf - 1:
                print('You have specified more scales than Number if filters minus 1.')

        g = [lambda x: kernel_simple_tf(t[0] * x, 'sf')]

        for i in range(Nf-1):
            g.append(lambda x, ind=i: kernel_simple_tf(t[ind] * x, 'wavelet'))

        self.g = g


class WarpedTranslates(Filter):
    r"""
    Creates a vertex frequency filterbank

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters (default = #TODO)

    Returns
    -------
    out : WarpedTranslates

    See :cite:`shuman2013spectrum`

    """

    def __init__(self, G, Nf, **kwargs):
        super(WarpedTranslat, self).__init__(G, **kwargs)
        raise NotImplementedError


class Papadakis(Filter):
    r"""
    Papadakis Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    a : float
        See equation above TODO for this parameter
        The spectrum is scaled between 0 and 2 (default = 3/4)

    Returns
    -------
    out : Papadakis

    """
    def __init__(self, G, a=0.75, **kwargs):
        super(Papadakis, self).__init__(G, **kwargs)

        g = [lambda x: papadakis(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1 - (papadakis(x*(2./G.lmax), a)) **
                                   2)))

        self.g = g

        def papadakis(val, a):
            y = np.empty(np.shape(val))
            l1 = a
            l2 = a*5./3

            r1ind = (val >= 0) * (val < l1)
            r2ind = (val >= l1) * (val < l2)
            r3ind = val >= l2

            y[r1ind] = 1
            y[r2ind] = np.sqrt((1 - np.sin(3*pi/(2*a) * val[r2ind]))/2.)
            y[r3ind] = 0

            return y


class Regular(Filter):
    r"""
    Regular Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    d : float
        See equation above TODO for this parameter
        Degree (default = 3)

    Returns
    -------
    out : Regular

    """
    def __init__(self, G, d=3, **kwargs):
        super(Regular, self).__init__(G, **kwargs)

        g = [lambda x: regular(x * (2./G.lmax), d)]
        g.append(lambda x: np.real(np.sqrt(1 - (regular(x * (2./G.lmax), d))
                                           ** 2)))

        self.g = g

        def regular(val, d):
            if d == 0:
                return np.sin(pi / 4.*val)

            else:
                output = np.sin(pi*(val - 1) / 2.)
                for i in range(2, d):
                    output = np.sin(pi*output / 2.)

                return np.sin(pi / 4.*(1 + output))


class Simoncelli(Filter):
    r"""
    Simoncelli Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    a : float
        See equation above TODO for this parameter
        The spectrum is scaled between 0 and 2 (default = 2/3)

    Returns
    -------
    out : Simoncelli

    """

    def __init__(self, G, a=2./3, **kwargs):
        super(Simoncelli, self).__init__(G, **kwargs)

        g = [lambda x: simoncelli(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1 -
                                           (simoncelli(x*(2./G.lmax), a))
                                           ** 2)))

        self.g = g

        def simoncelli(val, a):
            y = np.empty(np.shape(val))
            l1 = a
            l2 = 2 * a

            r1ind = (val >= 0) * (val < l1)
            r2ind = (val >= l1) * (val < l2)
            r3ind = (val >= l2)

            y[r1ind] = 1
            y[r2ind] = np.cos(pi/2 * np.log(val[r2ind]/float(a)) / np.log(2))
            y[r3ind] = 0

            return y


class Held(Filter):
    r"""
    Held Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    a : float
        See equation above TODO for this parameter
        The spectrum is scaled between 0 and 2 (default = 2/3)

    Returns
    -------
    out : Held

    """

    def __init__(self, G, a=2./3, **kwargs):
        super(Held, self).__init__(G, **kwargs)

        g = [lambda x: held(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1-(held(x * (2./G.lmax), a))
                                           ** 2)))

        self.g = g

        def held(val, a):
            y = np.empty(np.shape(val))
            l1 = a
            l2 = 2 * a
            mu = lambda x: -1. + 24.*x - 144.*x**2 + 256*x**3

            r1ind = (val >= 0) * (val < l1)
            r2ind = (val >= l1) * (val < l2)
            r3ind = (val >= l2)

            y[r1ind] = 1
            y[r2ind] = np.sin(2*pi*mu(val[r2ind]/(8.*a)))
            y[r3ind] = 0

            return y


class Heat(Filter):
    r"""
    Heat Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    tau : int or list of ints
        Scaling parameter. (default = 10)
    normalize (bool) : Normalize the kernel (works only if the eigenvalues are present in the graph)
        Default is 0

    Returns
    -------
    out : Heat

    """

    def __init__(self, G, tau=10, normalize=True, **kwargs):
        super(Heat, self).__init__(G, **kwargs)

        g = []

        if normalize:
            if not hasattr(G, 'e'):
                print('Filter Heat will calculate and set the eigenvalues to normalize the kernel')
                operators.compute_fourier_basis(G)

            if isinstance(tau, list):
                for t in tau:
                    gu = lambda x, taulam=t: np.exp(-taulam * x/G.lmax)
                    ng = linalg.norm(gu(G.e))
                    g.append(lambda x, taulam=t: np.exp(-taulam * x/G.lmax / ng))
            else:
                gu = lambda x: np.exp(-tau * x/G.lmax)
                ng = linalg.norm(gu(G.e))
                g.append(lambda x: np.exp(-tau * x/G.lmax / ng))

        else:
            if isinstance(tau, list):
                for t in tau:
                    g.append(lambda x, taulam=t: np.exp(-taulam * x/G.lmax))
            else:
                g.append(lambda x: np.exp(-tau * x/G.lmax))

        self.g = g


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a : int
        Description.
    b : array_like
        Description.
    c : bool
        Description.

    Returns
    -------
    d : ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.filters.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
