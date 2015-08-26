# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


class Regular(Filter):
    r"""
    Regular Filterbank

    Inherits its methods from Filters

    This function creates a parseval filterbank :math:`2` filters.
    The low-pass filter is defined by a function :math:`f_l(x)`
    between :math:`0` and :math:`2`. For :math:`d = 0`.

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} x \right)

    For :math:`d = 1`

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} \left( 1+ \sin\left(\frac{\pi}{2}(x-1)\right) \right) \right)

    For :math:`d = 2`

    .. math:: f_{l}= \sin\left( \frac{\pi}{4} \left( 1+ \sin\left(\frac{\pi}{2} \sin\left(\frac{\pi}{2}(x-1)\right)\right) \right) \right)

    And so for other degrees :math:`d`

    The high pass filter is adaptated to obtain a tight frame.

    Parameters
    ----------
    G : Graph
    d : float
        See equations above for this parameter
        Degree (default = 3)

    Returns
    -------
    out : Regular

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Regular(G)

    """
    def __init__(self, G, d=3, **kwargs):
        super(Regular, self).__init__(G, **kwargs)

        g = [lambda x: regular(x * (2./G.lmax), d)]
        g.append(lambda x: np.real(np.sqrt(1 - (regular(x * (2./G.lmax), d))
                                           ** 2)))

        self.g = g

        def regular(val, d):
            if d == 0:
                return np.sin(pi / 4.*val)

            else:
                output = np.sin(pi*(val - 1) / 2.)
                for i in range(2, d):
                    output = np.sin(pi*output / 2.)

                return np.sin(pi / 4.*(1 + output))
