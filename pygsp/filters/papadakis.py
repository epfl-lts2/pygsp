# -*- coding: utf-8 -*-

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
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """
    def __init__(self, G, a=0.75, **kwargs):

        g = [lambda x: papadakis(x * (2./G.lmax), a)]
        g.append(lambda x: np.real(np.sqrt(1 - (papadakis(x*(2./G.lmax), a)) **
                                   2)))

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

        super(Papadakis, self).__init__(G, g, **kwargs)
