# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Rectangular(Filter):
    r"""Design a rectangular filter.

    The filter evaluates at one in the interval [band_min, band_max] and zero
    everywhere else.

    Parameters
    ----------
    G : graph
    band_min : float
        Minimum relative band. The filter evaluates at 1 at this frequency.
        Zero corresponds to the smallest eigenvalue (which is itself equal to
        zero), one corresponds to the largest eigenvalue.
        If None, the filter is high-pass.
    band_max : float
        Maximum relative band. The filter evaluates at 1 at this frequency.
        If None, the filter is low-pass.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Rectangular(G, band_min=0.1, band_max=0.5)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, band_min=None, band_max=0.2):

        self.band_min = band_min
        self.band_max = band_max

        def kernel_lowpass(x):
            x = x / G.lmax
            return x <= band_max

        def kernel_highpass(x):
            x = x / G.lmax
            return x >= band_min

        if (band_min is None) and (band_max is None):
            kernel = lambda x: np.ones_like(x)
        elif band_min is None:
            kernel = kernel_lowpass
        elif band_max is None:
            kernel = kernel_highpass
        else:
            kernel = lambda x: kernel_lowpass(x) * kernel_highpass(x)

        super(Rectangular, self).__init__(G, kernel)

    def _get_extra_repr(self):
        attrs = dict()
        if self.band_min is not None:
            attrs.update(band_min='{:.2f}'.format(self.band_min))
        if self.band_max is not None:
            attrs.update(band_max='{:.2f}'.format(self.band_max))
        return attrs
