# -*- coding: utf-8 -*-

import numpy as np
from . import Filter
from math import pi


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

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Held(G)

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
