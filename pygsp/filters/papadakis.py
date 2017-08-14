# -*- coding: utf-8 -*-

import numpy as np

from . import Filter


class Papadakis(Filter):
    r"""
    Papadakis filterbank

    This function create a parseval filterbank of :math:`2`.
    The low-pass filter is defined by the function

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\
            \sqrt{1-\frac{\sin\left(\frac{3\pi}{2a}x\right)}{2}} & \mbox{if }a<x\leq \frac{5a}{3} \\
            0 & \mbox{if }x>\frac{5a}{3} \end{cases}

    The high pass filter is adapted to obtain a tight frame.

    Parameters
    ----------
    G : graph
    a : float
        See above equation for this parameter.
        The spectrum is scaled between 0 and 2 (default = 3/4).

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
            y[r2ind] = np.sqrt((1 - np.sin(3*np.pi/(2*a) * val[r2ind]))/2.)
            y[r3ind] = 0

            return y
