# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


class Held(Filter):
    r"""
    Held Filterbank

    Inherits its methods from Filters

    This function create a parseval filterbank of :math:`2` filters.
    The low-pass filter is defined by a function :math:`f_l(x)`

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\ \sin\left(2\pi\mu\left(\frac{x}{8a}\right)\right) & \mbox{if }a<x\leq2a\\ 0 & \mbox{if }x>2a \end{cases}

    with

    .. math:: \mu(x) = -1+24x-144*x^2+256*x^3

    The high pass filter is adaptated to obtain a tight frame.

    Parameters
    ----------
    G : Graph
    a : float
        See equation above for this parameter
        The spectrum is scaled between 0 and 2 (default = 2/3)

    Returns
    -------
    out : Held

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Held(G)

    """

    def __init__(self, G, a=2./3, **kwargs):
        super(Held, self).__init__(G, **kwargs)

        g = [lambda x: held(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1 - (held(x * (2./G.lmax), a))
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
