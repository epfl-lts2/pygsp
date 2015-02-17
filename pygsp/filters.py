# -*- coding: utf-8 -*-

r"""
Flters Doc
"""

import numpy as np


class Filter(object):
    r"""
    TODO doc
    """

    def __init__(self):
        pass

    def analysis(G, s, **kwargs):
        Nf = len(self.fi)
        pass

    def evaluate(x):
        pass

    def inverse(G, c, **kwargs):
        pass

    def synthesis(G, c, **kwargs):
        pass

    def approx(G, m, N, **kwargs):
        pass

    def tighten(G):
        pass

    def bank_bounds(G, **kwargs):
        pass

    def bank_matrix(G, **kwargs):
        pass

    def wlog_scales(lim, lmax, Fscales):
        pass

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

    def __init__(self):
        pass


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
