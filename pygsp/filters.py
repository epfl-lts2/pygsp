# -*- coding: utf-8 -*-

r"""
Flters Doc
"""

from math import exp, log
import numpy as np

from pygsp import utils


class Filter(object):
    r"""
    TODO doc
    """

    def __init__(self, **kwargs):
        pass

    def analysis(self, G, s, **kwargs):
        Nf = len(self.fi)
        pass

    def evaluate(self, x):
        pass

    def inverse(self, G, c, **kwargs):
        pass

    def synthesis(self, G, c, **kwargs):
        pass

    def approx(G, m, N, **kwargs):
        pass

    def tighten(G):
        pass

    def bank_bounds(G, **kwargs):
        pass

    def bank_matrix(G, **kwargs):
        pass

    def wlog_scales(self, lmin, lmax, Nscales, t1=1, t2=2):
        r"""
        Compute logarithm scales for wavelets
        """
        print(lmin)
        smin = t1/lmax
        smax = t2/lmin

        s = np.exp(np.linspace(log(smax), log(smin), Nscales))

        return s

    def evaluate_can_dual(val):
        pass

    def can_dual():
        pass

    def vec2mat(d, Nf):
        pass

    def mat2vec(d):
        pass


class FilterBank(Filter):
    r"""
    A filterbank should just be a list of filter to apply
    """

    def __init__(self, F):
        self.F = F


class Abspline(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class Expwin(Filter):

    def __init__(self, G, bmax, a):
        pass


class HalfCosine(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class Itersine(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class MexicanHat(Filter):

    def __init__(self, G, Nf=6, lpfactor=20, t=None, **kwargs):
        try:
            G.lmax
        except AttributeError:
            G.lmax = utils.estimate_lmax(G)
        print(G.lmax)

        if t is None:
            G.lmin = G.lmax / lpfactor
            print(G.lmin)
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: x * np.exp(-x)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        self.g = []
        print(self.g)
        self.g.append(lambda x: 1.2 * exp(-1) * gl(x / lminfac))

        for i in range(1, Nf):
            self.g.append(lambda x: gb(self.t[i] * x))


class Meyer(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class SimpleTf(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class WarpedTranslat(Filter):

    def __init__(self, G, Nf, **kwargs):
        pass


class Papadakis(Filter):

    def __init__(self, G, **kwargs):
        pass


class Regular(Filter):

    def __init__(self, G, **kwargs):
        pass


class Simoncelli(Filter):

    def __init__(self, G, **kwargs):
        pass


class Held(Filter):

    def __init__(self, G, **kwargs):
        pass


class Heat(Filter):

    def __init__(self, G, tau, **kwargs):
        pass


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
