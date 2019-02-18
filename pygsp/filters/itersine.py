# -*- coding: utf-8 -*-

from __future__ import division

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
    >>> g = filters.Itersine(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, overlap=2):

        self.overlap = overlap
        self.mu = np.linspace(0, G.lmax, num=Nf)

        scales = G.lmax / (Nf - overlap + 1) * overlap

        def kernel(x):
            y = np.cos(x * np.pi)**2
            y = np.sin(0.5 * np.pi * y)
            return y * ((x >= -0.5) * (x <= 0.5))

        kernels = []
        for i in range(1, Nf + 1):

            def kernel_centered(x, i=i):
                y = kernel(x / scales - (i - overlap / 2) / overlap)
                return y * np.sqrt(2 / overlap)

            kernels.append(kernel_centered)

        super(Itersine, self).__init__(G, kernels)

    def _get_extra_repr(self):
        return dict(overlap='{:.2f}'.format(self.overlap))
