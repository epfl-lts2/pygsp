# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


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

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Papadakis(G)

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
