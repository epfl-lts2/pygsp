# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Expwin(Filter):
    r"""Design an exponential window filter.

    Parameters
    ----------
    G : graph
    band_max : float
        Maximum relative band (default = 0.2)
    slope : float
        Slope parameter (default = 1)

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Expwin(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, band_max=0.2, slope=1):

        def fx(x):
            # Canary to avoid division by zero and overflow.
            y = np.where(x <= 0, -1, x)
            y = np.exp(-slope / y)
            return np.where(x <= 0, 0, y)

        def gx(x):
            y = fx(x)
            return y / (y + fx(1 - x))

        def ffin(x):
            return gx(1 - x)

        g = [lambda x: ffin(x/band_max/G.lmax)]

        super(Expwin, self).__init__(G, g)
