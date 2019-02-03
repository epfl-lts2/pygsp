# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Papadakis(Filter):
    r"""Design 2 filters with the Papadakis construction (tight frame).

    This function creates a Parseval filter bank of 2 filters.
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

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Papadakis(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, a=0.75):

        self.a = a

        def kernel(x, a):
            y = np.empty(np.shape(x))
            l1 = a
            l2 = a * 5 / 3

            r1ind = (x >= 0) * (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2)

            y[r1ind] = 1
            y[r2ind] = np.sqrt((1 - np.sin(3*np.pi/(2*a) * x[r2ind]))/2)
            y[r3ind] = 0

            return y

        papadakis = Filter(G, lambda x: kernel(x*2/G.lmax, a))
        complement = papadakis.complement(frame_bound=1)
        kernels = papadakis._kernels + complement._kernels

        super(Papadakis, self).__init__(G, kernels)

    def _get_extra_repr(self):
        return dict(a='{:.2f}'.format(self.a))
