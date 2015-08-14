# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


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

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Regular(G)

    """
    def __init__(self, G, d=3, **kwargs):
        super(Regular, self).__init__(G, **kwargs)

        g = [lambda x: regular(x * (2./G.lmax), d)]
        g.append(lambda x: np.real(np.sqrt(1 - (regular(x * (2./G.lmax), d))
                                           ** 2)))

        self.g = g

        def regular(val, d):
            if d == 0:
                return np.sin(pi / 4.*val)

            else:
                output = np.sin(pi*(val - 1) / 2.)
                for i in range(2, d):
                    output = np.sin(pi*output / 2.)

                return np.sin(pi / 4.*(1 + output))
