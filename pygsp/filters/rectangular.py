# -*- coding: utf-8 -*-

from __future__ import division

from . import Filter  # prevent circular import in Python < 3.5


class Rectangular(Filter):
    r"""Design a rectangular filter.

    Parameters
    ----------
    G : graph
    band_min : float
        Minimum relative band. The filter take the value 0.5 at this relative
        frequency (default = None).
    band_max : float
        Maximum relative band. The filter take the value 0.5 at this relative
        frequency (default = 0.2).

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
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, band_min=None, band_max=0.2):

        if band_min is None:
            band_min = 0
        if band_max is None:
            band_max = 1

        if not 0 <= band_min <= 1:
            raise ValueError('band_min should be in [0, 1]')
        if not 0 <= band_max <= 1:
            raise ValueError('band_max should be in [0, 1]')
        if not band_max >= band_min:
            raise ValueError('band_max should be greater than band_min')

        self.band_min = band_min
        self.band_max = band_max

        def kernel(x):
            x = x / G.lmax
            return (x >= band_min) & (x <= band_max)

        super(Rectangular, self).__init__(G, [kernel])

    def _get_extra_repr(self):
        return dict(band_min='{:.2f}'.format(self.band_min),
                    band_max='{:.2f}'.format(self.band_max))
