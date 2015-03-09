# -*- coding: utf-8 -*-

r"""
Filters Doc
"""

from math import exp, log
import numpy as np
import scipy as sp

from pygsp import utils, operators


class Filter(object):
    r"""
    TODO doc
    """

    def __init__(self, **kwargs):
        pass

    def analysis(self, G, s, exact=True, cheb_order=30, **kwargs):
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

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class Expwin(Filter):

    def __init__(self, G, bmax, a):
        raise NotImplementedError


class HalfCosine(Filter):

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class Itersine(Filter):

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class MexicanHat(Filter):

    def __init__(self, G, Nf=6, lpfactor=20, t=None, normalize=False,
                 **kwargs):

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

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class SimpleTf(Filter):

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class WarpedTranslat(Filter):

    def __init__(self, G, Nf, **kwargs):
        raise NotImplementedError


class Papadakis(Filter):

    def __init__(self, G, **kwargs):
        raise NotImplementedError


class Regular(Filter):

    def __init__(self, G, **kwargs):
        raise NotImplementedError


class Simoncelli(Filter):

    def __init__(self, G, **kwargs):
        raise NotImplementedError


class Held(Filter):

    def __init__(self, G, **kwargs):
        raise NotImplementedError


class Heat(Filter):

    def __init__(self, G, tau, **kwargs):
        raise NotImplementedError


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
