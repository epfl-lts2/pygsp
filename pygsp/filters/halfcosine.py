# -*- coding: utf-8 -*-

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class HalfCosine(Filter):
    r"""Design an half cosine filter bank (tight frame).

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.HalfCosine(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, **kwargs):

        if Nf <= 2:
            raise ValueError('The number of filters must be higher than 2.')

        dila_fact = G.lmax * (3./(Nf - 2))

        main_window = lambda x: np.multiply(np.multiply((.5 + .5*np.cos(2.*np.pi*(x/dila_fact - 1./2))), (x >= 0)), (x <= dila_fact))

        g = []

        for i in range(Nf):
            g.append(lambda x, ind=i: main_window(x - dila_fact/3. * (ind - 2)))

        super(HalfCosine, self).__init__(G, g, **kwargs)
