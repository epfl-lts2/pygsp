# -*- coding: utf-8 -*-

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Expwin(Filter):
    r"""Design an exponential window filter.

    Parameters
    ----------
    G : graph
    bmax : float
        Maximum relative band (default = 0.2)
    a : int
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

    def __init__(self, G, bmax=0.2, a=1., **kwargs):

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

        super(Expwin, self).__init__(G, g, **kwargs)
