# -*- coding: utf-8 -*-

from . import Filter

import numpy as np


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

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Expwin(G)

    """
    def __init__(self, G, bmax=0.2, a=1., **kwargs):
        super(Expwin, self).__init__(G, **kwargs)

        def fx(x, a):
            y = np.exp(-float(a)/x)
            if isinstance(x, np.ndarray):
                y = np.where(x <= 0, 0., y)
            else:
                if x <= 0:
                    y = 0.
            return y

        def gx(x, a):
            y = fx(x, a)
            return y/(y + fx(1 - x, a))

        def ffin(x, a):
            return gx(1 - x, a)

        g = [lambda x: ffin(np.float64(x)/bmax/G.lmax, a)]
        self.g = g
