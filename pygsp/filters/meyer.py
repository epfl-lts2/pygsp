# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


class Meyer(Filter):
    r"""
    Meyer Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)

    Returns
    -------
    out : Meyer

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Meyer(G)

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(Meyer, self).__init__(G, **kwargs)

        if not hasattr(G, 't'):
            G.t = (4./(3 * G.lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(G.t) >= Nf - 1:
            self.logger.warning('You have specified more scales than'
                                ' the number of scales minus 1')

        t = G.t

        g = [lambda x: kernel_meyer(t[0] * x, 'sf')]
        for i in range(Nf - 1):
            g.append(lambda x, ind=i: kernel_meyer(t[ind] * x, 'wavelet'))

        self.g = g

        def kernel_meyer(x, kerneltype):
            r"""
            Evaluates Meyer function and scaling function

            Parameters
            ----------
            x : ndarray
                Array of independant variables values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns
            -------
            r : ndarray

            """

            x = np.array(x)

            l1 = 2/3.
            l2 = 4/3.
            l3 = 8/3.

            v = lambda x: x ** 4. * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1)*(x < l2)
            r3ind = (x >= l2)*(x < l3)

            r = np.empty(x.shape)
            if kerneltype is 'sf':
                r[r1ind] = 1
                r[r2ind] = np.cos((pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kerneltype is 'wavelet':
                r[r2ind] = np.sin((pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise TypeError('Unknown kernel type ', kerneltype)

            return r
