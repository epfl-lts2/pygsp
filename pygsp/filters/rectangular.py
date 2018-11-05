# -*- coding: utf-8 -*-

from __future__ import division

from . import Filter  # prevent circular import in Python < 3.5


class Rectangular(Filter):
    r"""Design a rectangular filter.

    The filter evaluates at one in the interval [band_min, band_max] and zero
    everywhere else.

    Parameters
    ----------
    G : graph
    band_min : float
        Minimum relative band, a number in [0, 1]. Zero corresponds to the
        smallest eigenvalue (which is itself equal to zero), one corresponds to
        the largest eigenvalue.
    band_max : float
        Maximum relative band, a number in [0, 1].

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

    def __init__(self, G, band_min=0, band_max=0.2):

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
