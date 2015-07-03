# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


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

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.HalfCosine(G)

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(HalfCosine, self).__init__(G, **kwargs)

        if Nf <= 2:
            raise ValueError('The number of filters must be higher than 2.')

        dila_fact = G.lmax * (3./(Nf - 2))

        main_window = lambda x: np.multiply(np.multiply((.5 + .5*np.cos(2.*pi*(x/dila_fact - 1./2))), (x >= 0)), (x <= dila_fact))

        g = []

        for i in range(Nf):
            g.append(lambda x, ind=i: main_window(x - dila_fact/3. * (ind - 2)))

        self.g = g
