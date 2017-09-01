# -*- coding: utf-8 -*-

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Itersine(Filter):
    r"""Design an itersine filter bank (tight frame).

    Create an itersine half overlap filter bank of Nf filters.
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
    def __init__(self, G, Nf=6, overlap=2., **kwargs):

        def k(x):
            return np.sin(0.5*np.pi*np.power(np.cos(x*np.pi), 2)) * ((x >= -0.5)*(x <= 0.5))

        scale = G.lmax/(Nf - overlap + 1.)*overlap
        g = []

        for i in range(1, Nf + 1):
            g.append(lambda x, ind=i: k(x/scale - (ind - overlap/2.)/overlap) / np.sqrt(overlap)*np.sqrt(2))

        super(Itersine, self).__init__(G, g, **kwargs)
