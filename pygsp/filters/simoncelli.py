# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


class Simoncelli(Filter):
    r"""
    Simoncelli Filterbank

    Inherits its methods from Filters

    This function create a parseval filterbank of :math:`2`.
    The low-pass filter is defined by a function :math:`f_l(x)`.

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\ \cos\left(\frac{\pi}{2}\frac{\log\left(\frac{x}{2}\right)}{\log(2)}\right) & \mbox{if }a<x\leq2a\\ 0 & \mbox{if }x>2a \end{cases}

    The high pass filter is is adaptated to obtain a tight frame.

    Parameters
    ----------
    G : Graph
    a : float
        See equation above for this parameter
        The spectrum is scaled between 0 and 2 (default = 2/3)

    Returns
    -------
    out : Simoncelli

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Simoncelli(G)

    """

    def __init__(self, G, a=2./3, **kwargs):
        super(Simoncelli, self).__init__(G, **kwargs)

        g = [lambda x: simoncelli(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1 -
                                           (simoncelli(x*(2./G.lmax), a))
                                           ** 2)))

        self.g = g

        def simoncelli(val, a):
            y = np.empty(np.shape(val))
            l1 = a
            l2 = 2 * a

            r1ind = (val >= 0) * (val < l1)
            r2ind = (val >= l1) * (val < l2)
            r3ind = (val >= l2)

            y[r1ind] = 1
            y[r2ind] = np.cos(pi/2 * np.log(val[r2ind]/float(a)) / np.log(2))
            y[r3ind] = 0

            return y
