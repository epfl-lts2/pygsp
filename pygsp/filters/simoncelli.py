# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Simoncelli(Filter):
    r"""Design 2 filters with the Simoncelli construction (tight frame).

    This function creates a Parseval filter bank of 2 filters.
    The low-pass filter is defined by the function

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\
            \cos\left(\frac{\pi}{2}\frac{\log\left(\frac{x}{2}\right)}{\log(2)}\right) & \mbox{if }a<x\leq2a\\
            0 & \mbox{if }x>2a \end{cases}

    The high pass filter is adapted to obtain a tight frame.

    Parameters
    ----------
    G : graph
    a : float
        See above equation for this parameter.
        The spectrum is scaled between 0 and 2 (default = 2/3).

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Simoncelli(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, a=2/3):

        self.a = a

        def kernel(x, a):
            y = np.empty(np.shape(x))
            l1 = a
            l2 = 2 * a

            r1ind = (x >= 0) * (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2)

            y[r1ind] = 1
            y[r2ind] = np.cos(np.pi/2 * np.log(x[r2ind]/a) / np.log(2))
            y[r3ind] = 0

            return y

        simoncelli = Filter(G, lambda x: kernel(x*2/G.lmax, a))
        complement = simoncelli.complement(frame_bound=1)
        kernels = simoncelli._kernels + complement._kernels

        super(Simoncelli, self).__init__(G, kernels)

    def _get_extra_repr(self):
        return dict(a='{:.2f}'.format(self.a))
