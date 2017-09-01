# -*- coding: utf-8 -*-

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Regular(Filter):
    r"""Design 2 filters with the regular construction (tight frame).

    This function creates a Parseval filter bank of 2 filters.
    The low-pass filter is defined by a function :math:`f_l(x)`
    between :math:`0` and :math:`2`. For :math:`d = 0`.

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} x \right)

    For :math:`d = 1`

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} \left( 1+ \sin\left(\frac{\pi}{2}(x-1)\right) \right) \right)

    For :math:`d = 2`

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} \left( 1+ \sin\left(\frac{\pi}{2} \sin\left(\frac{\pi}{2}(x-1)\right)\right) \right) \right)

    And so forth for other degrees :math:`d`.

    The high pass filter is adapted to obtain a tight frame.

    Parameters
    ----------
    G : graph
    d : float
        Degree (default = 3). See above equations.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Regular(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """
    def __init__(self, G, d=3, **kwargs):

        g = [lambda x: regular(x * (2./G.lmax), d)]
        g.append(lambda x: np.real(np.sqrt(1 - (regular(x * (2./G.lmax), d))
                                           ** 2)))

        def regular(val, d):
            if d == 0:
                return np.sin(np.pi / 4.*val)

            else:
                output = np.sin(np.pi*(val - 1) / 2.)
                for i in range(2, d):
                    output = np.sin(np.pi*output / 2.)

                return np.sin(np.pi / 4.*(1 + output))

        super(Regular, self).__init__(G, g, **kwargs)
