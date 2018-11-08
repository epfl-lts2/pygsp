# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Expwin(Filter):
    r"""Design an exponential window filter.

    The window has the shape of a (half) triangle with the corners smoothed by
    an exponential function.

    Parameters
    ----------
    G : graph
    band_min : float
        Minimum relative band. The filter evaluates at 0.5 at this frequency.
        Zero corresponds to the smallest eigenvalue (which is itself equal to
        zero), one corresponds to the largest eigenvalue.
        If None, the filter is high-pass.
    band_max : float
        Maximum relative band. The filter evaluates at 0.5 at this frequency.
        If None, the filter is low-pass.
    slope : float
        The slope at cut-off.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Expwin(G, band_min=0.1, band_max=0.7, slope=5)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, band_min=None, band_max=0.2, slope=1):

        self.band_min = band_min
        self.band_max = band_max
        self.slope = slope

        def exp(x):
            """Exponential function with canary to avoid division by zero and
            overflow."""
            y = np.where(x <= 0, -1, x)
            y = np.exp(-slope / y)
            return np.where(x <= 0, 0, y)

        def h(x):
            y = exp(x)
            z = exp(1 - x)
            return y / (y + z)

        def kernel_lowpass(x):
            return h(0.5 - x/G.lmax + band_max)
        def kernel_highpass(x):
            return h(0.5 + x/G.lmax - band_min)

        if (band_min is None) and (band_max is None):
            kernel = lambda x: np.ones_like(x)
        elif band_min is None:
            kernel = kernel_lowpass
        elif band_max is None:
            kernel = kernel_highpass
        else:
            kernel = lambda x: kernel_lowpass(x) * kernel_highpass(x)

        super(Expwin, self).__init__(G, kernel)

    def _get_extra_repr(self):
        attrs = dict()
        if self.band_min is not None:
            attrs.update(band_min='{:.2f}'.format(self.band_min))
        if self.band_max is not None:
            attrs.update(band_max='{:.2f}'.format(self.band_max))
        attrs.update(slope='{:.0f}'.format(self.slope))
        return attrs
