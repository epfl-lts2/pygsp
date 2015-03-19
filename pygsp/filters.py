# -*- coding: utf-8 -*-

r"""
Filters Doc
"""

from math import exp, log, pi
import numpy as np
from numpy import linalg
import scipy as sp
import scipy.optimize
import math

from pygsp import utils, operators


class Filter(object):
    r"""
    Parent class for all Filters or Filterbanks, contains the shared
    methods for those classes.
    """

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose

    def analysis(self, G, s, exact=True, cheb_order=30, **kwargs):
        r"""
        Operator to analyse a filterbank

        Parameters
        ----------
        G : Graph object
        s : ndarray
            graph signal to analyse
        exact : bool
            wether using an exact method or cheby approx
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
        >>> sen = graphs.Sensor()
        >>> MH = filters.MexicanHat(sen)
        >>> x = np.random.rand(64, 64)
        >>> co = MH.analyse(sen, x)

        """
        Nf = len(self.g)

        if exact:
            if not hasattr(G, 'e') or not hasattr(G, 'U'):
                G = operators.compute_fourier_basis(G)
            Nv = s.shape[1]
            c = np.zeros((G.N * Nf, Nv))

            fie = self.evaluate(G.e)

            for i in range(Nf):
                c[np.arange(G.N) + G.N * (i-1)] =\
                    operators.igft(G, sp.kron(sp.ones((fie[0][i], 1)), Nv) *
                                   operators.gft(G, s))

        else:  # Chebyshev approx
            if not hasattr(G, 'lmax'):
                G = utils.estimate_lmax(G)

            cheb_coef = operators.compute_cheby_coeff(self.g, G, m=cheb_order)
            c = operators.cheby_op(G, cheb_coef, s)

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
        >>> sen = graphs.Sensor()
        >>> MH = filters.MexicanHat(sen)
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

    def synthesis(self, G, c, **kwargs):
        raise NotImplementedError

    def approx(G, m, N, **kwargs):
        raise NotImplementedError

    def tighten(G):
        raise NotImplementedError

    def bank_bounds(G, **kwargs):
        raise NotImplementedError

    def bank_matrix(G, **kwargs):
        raise NotImplementedError

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

    def evaluate_can_dual(val):
        raise NotImplementedError

    def can_dual():
        pass
        raise NotImplementedError

    def vec2mat(d, Nf):
        raise NotImplementedError

    def mat2vec(d):
        raise NotImplementedError


class FilterBank(Filter):
    r"""
    A filterbank should just be a list of filter to apply
    """

    def __init__(self, F):
        self.F = F


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
    t = ndarray
        Vector of scale to be used (Initialized by default at
        the value of the log scale)

    Returns
    -------
    out : Abspline

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, **kwargs):
        super(Abspline, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        G.lmin = G.lmax / lpfactor

        if t is None:
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: utils.kernel_abspline3(x, 2, 2, 1, 2)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        self.g = []
        self.g.append(lambda x: 1.2 * exp(-1) * gl(x / lminfac))

        for i in range(0, Nf-1):
            self.g.append(lambda x, ind=i: gb(self.t[ind] * x))

        f = lambda x: -gb(x)
        xstar = scipy.optimize.minimize_scalar(f, method='Bounded',
                                               bounds=(1, 2))
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
    bmax = float
        Maximum relative band (default = 0.2)
    a = int
        Slope parameter (default = 1)

    Returns
    -------
    out : Expwin

    """
    def __init__(self, G, bmax=0.2, a=1, **kwargs):
        super(Expwin, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        def fx(x, a):
            y = np.exp(-a / x)
            for val, ind in y:
                if val < 0:
                    y[ind] = 0
            return y

        def gx(x, a):
            y = fx(x, a)
            return y / (y + fx(1 - x, a))

        ffin = lambda x, a: gx(1 - x, a)

        g = lambda x: ffin(x/bmax/G.lmax, a)

        self.g = g


class HalfCosine(Filter):
    r"""
    HalfCosine Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf = int
        Number of filters from 0 to lmax

    Returns
    -------
    out : HalfCosine

    """

    def __init__(self, G, Nf, **kwargs):
        super(HalfCosine, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        dila_fact = G.lmax * (3/(Nf - 2))

        main_window = lambda x: (.5 + .5 * np.cos(2. * pi * (x/dila_fact - 1/2))) *\
                                (x >= 0) * (x <= dila_fact)

        g = []

        for i in range(Nf):
            g.append(lambda x, ind=i: main_window(x - dila_fact/3 * (ind-3)))


class Itersine(Filter):

    def __init__(self, G, Nf, **kwargs):
        super(Itersine, self).__init__(**kwargs)
        raise NotImplementedError


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
    t = ndarray
        Vector of scale to be used (Initialized by default at
        the value of the log scale)
    normalize : bool
        Wether to normalize the wavelet by the factor/sqrt(t) (default = False)

    Returns
    -------
    out : MexicanHat

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, normalize=False,
                 **kwargs):
        super(MexicanHat, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        if t is None:
            G.lmin = G.lmax / lpfactor
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: x * np.exp(-x)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        self.g = []
        self.g.append(lambda x: 1.2 * exp(-1) * gl(x / lminfac))

        for i in range(0, Nf-1):
            if normalize:
                self.g.append(lambda x, ind=i: np.sqrt(t[i]) *
                              gb(self.t[ind] * x))
            else:
                self.g.append(lambda x, ind=i: gb(self.t[ind] * x))


class Meyer(Filter):
    r"""
    Meyer Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf = int
        Number of filters from 0 to lmax (default = 6)

    Returns
    -------
    out : Meyer

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(Meyer, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        if not hasattr(G, 't'):
            G.t = (4/(3 * G.lmax)) * np.power(2., [Nf-2, -1, 0])

        if len(G.t) >= Nf-1:
            print('GSP_KERNEL_MEYER: You have specified more scales than\
                  the number of scales minus 1')

        t = G.t
        g = []

        g.append(lambda x: operators.kernel_meyer(t[1] * x, 'sf'))
        for i in range(Nf-1):
            g.append(lambda x, ind=i: operators.kernel_meyer(t[ind] * x,
                                                             'wavelet'))

        self.g = g


class SimpleTf(Filter):
    r"""
    SimpleTf Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf = int
        Number of filters from 0 to lmax
    t = ndarray
        Vector of scale to be used (Initialized by default at
        the value of the log scale)

    Returns
    -------
    out : SimpleTf

    """

    def __init__(self, G, Nf, t=None, **kwargs):
        super(SimpleTf, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            G.lmax = utils.estimate_lmax(G)

        if not t:
            t = (1./(2. * G.lmax) * 2. ** (range(0, Nf-2, -1)))

        if self.verbose:
            if len(t) >= Nf - 1:
                print('GSP_SIMPLETF: You have specified more scales than Number\
                      if filters minus 1')

        g = []

        g.append(lambda x: kernel_simple_tf(t(1) * x, 'sf'))
        for i in range(Nf-1):
            g.append(lambda x, ind=i: kernel_simple_tf(t[i] * x, 'wavelet'))


class WarpedTranslat(Filter):

    def __init__(self, G, Nf, **kwargs):
        super(WarpedTranslat, self).__init__(**kwargs)
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

    def __init__(self, G, a=3/4, **kwargs):
        super(Papadakis, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            if self.verbose:
                print('GSP_PAPADAKIS: has to compute lmax')
            G = utils.estimate_lmax(G)

        g = []
        g.append(lambda x: papadakis(x * (2/G.lmax), a))
        g.append(lambda x: np.real(np.sqrt(1-(papadakis(x * 2/G.lmax), a)) **
                                   2))

        def papadakis(val, a):
            y = []
            l1 = a
            l2 = 2 * a/3

            r1ind = np.extract(val >= 0 and val < l1)
            r2ind = np.extract(val >= l1 and val < l2)
            r3ind = np.extract(val >= l2)

            y[r1ind] = 1
            y[r2ind] = np.sqrt((1 - np.sin(3 * pi/(2 * a) * val[r2ind]))/2)
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
        super(Regular, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            if self.verbose:
                print('GSP_REGULAR: has to compute lmax')
            G = utils.estimate_lmax(G)

        g = []
        g.append(lambda x: regular(x * (2/G.lmax), d))
        g.append(lambda x: np.real(np.sqrt(1-(regular(x * (2/G.lmax), d))
                                           ** 2)))

        def regular(val, d):
            if d == 0:
                return np.sin(pi / 4 * val)
            else:
                output = np.sin(pi * (val - 1) / 2)
                for i in range(2, d):
                    output = np.sin(pi * output / 2)
                return np.sin(pi / 4 * (1 + output))


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

    def __init__(self, G, a=2/3, verbose=False, **kwargs):
        super(Simoncelli, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            if verbose:
                print('GSP_SIMONCELLI: has to compute lmax')
            G = utils.estimate_lmax(G)

        g = []
        g.append(lambda x: simoncelli(x * (2/G.lmax), a))
        g.append(lambda x: np.real(np.sqrt(1 -
                                           (simoncelli(x * (2/G.lmax), a))
                                           ** 2)))

        self.g = g

        def simoncelli(val, a):
            y = []
            l1 = a
            l2 = 2 * a

            r1ind = np.extract(val >= 0 and val < l1)
            r2ind = np.extract(val >= l1 and val < l2)
            r3ind = np.extract(val >= l2)

            y[r1ind] = 1
            y[r2ind] = np.cos(pi/2 * np.log(val[r2ind] / a) / np.log(2))
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

    def __init__(self, G, a=2/3, **kwargs):
        super(Held, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            if self.verbose:
                print('GSP_HELD: has to compute lmax')
            G = utils.estimate_lmax(G)

        g = []
        g.append(lambda x: self.held(x * (2/G.lmax), a))
        g.append(lambda x: np.real(np.sqrt(1-(self.held(x * (2/G.lmax), a))
                                           ** 2)))

    def held(val, a):
        y = []

        l1 = a
        l2 = 2 * a
        mu = lambda x: -1. + 24. * x - 144. * x ** 2 + 256 * x ** 3

        r1ind = np.extract(val >= 0 and val < l1)
        r2ind = np.extract(val >= l1 and val < l2)
        r3ind = np.extract(val >= l2)

        y[r1ind] = 1
        y[r2ind] = np.sin(2 * pi * mu(val[r2ind] / (8 * a)))
        y[r3ind] = 0

        return y


class Heat(Filter):
    r"""
    Heat Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    tau : int
        Scaling parameter (default = 10)
    normalize : bool
        Normalize the kernel (works only if the eigenvalues are
        present in the graph)
        (default = 0)

    Returns
    -------
    out : Heat

    """

    def __init__(self, G, tau=10, normalize=False, **kwargs):
        super(Heat, self).__init__(**kwargs)

        if not hasattr(G, 'lmax'):
            if self.verbose:
                print('GSP_HEAT: has to compute lmax')
            G = utils.estimate_lmax(G)

        if normalize:
            gu = lambda x: np.exp(-tau * x/G.lmax)
            ng = linalg.norm(gu(G.E))
            g = lambda x: np.exp(-tau * x/G.lmax / ng)

        else:
            g = lambda x: np.exp(-tau * x/G.lmax)

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
