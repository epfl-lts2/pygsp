# -*- coding: utf-8 -*-

from __future__ import division

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
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6):

        if Nf <= 2:
            raise ValueError('The number of filters must be greater than 2.')

        dila_fact = G.lmax * 3 / (Nf - 2)

        def kernel(x):
            y = np.cos(2 * np.pi * (x / dila_fact - .5))
            y = np.multiply((.5 + .5*y), (x >= 0))
            return np.multiply(y, (x <= dila_fact))

        kernels = []

        for i in range(Nf):

            def kernel_centered(x, i=i):
                return kernel(x - dila_fact/3 * (i - 2))

            kernels.append(kernel_centered)

        super(HalfCosine, self).__init__(G, kernels)
