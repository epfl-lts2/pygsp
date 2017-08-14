# -*- coding: utf-8 -*-

import numpy as np

from . import Filter


class Itersine(Filter):
    r"""
    Itersine filterbank

    Create an itersine half overlap filterbank of Nf filters.
    Going from 0 to lambda_max.

    Parameters
    ----------
    G : graph
    Nf : int (optional)
        Number of filters from 0 to lmax. (default = 6)
    overlap : int (optional)
        (default = 2)

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Itersine(G)

    """
    def __init__(self, G, Nf=6, overlap=2., **kwargs):
        super(Itersine, self).__init__(G, **kwargs)

        def k(x):
            return np.sin(0.5*np.pi*np.power(np.cos(x*np.pi), 2)) * ((x >= -0.5)*(x <= 0.5))

        scale = G.lmax/(Nf - overlap + 1.)*overlap
        g = []

        for i in range(1, Nf + 1):
            g.append(lambda x, ind=i: k(x/scale - (ind - overlap/2.)/overlap) / np.sqrt(overlap)*np.sqrt(2))

        self.g = g
