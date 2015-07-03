# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import pi


class SimpleTf(Filter):
    r"""
    SimpleTf Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    t : ndarray
        Vector of scale to be used (Initialized by default at the value
        of the log scale)

    Returns
    -------
    out : SimpleTf

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.SimpleTf(G)

    """

    def __init__(self, G, Nf=6, t=None, **kwargs):
        super(SimpleTf, self).__init__(G, **kwargs)

        def kernel_simple_tf(x, kerneltype):
            r"""
            Evaluates 'simple' tight-frame kernel

            Parameters
            ----------
            x : ndarray
                Array if independant variable values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns:
            r : ndarray

            """

            l1 = 0.25
            l2 = 0.5
            l3 = 1.

            h = lambda x: np.sin(pi*x/2.)**2

            r1ind = x < l1
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            r = np.zeros(x.shape)
            if kerneltype is 'sf':
                r[r1ind] = 1.
                r[r2ind] = np.sqrt(1 - h(4*x[r2ind] - 1)**2)
            elif kerneltype is 'wavelet':
                r[r2ind] = h(4*(x[r2ind] - 1/4.))
                r[r3ind] = np.sqrt(1 - h(2*x[r3ind] - 1)**2)
            else:
                raise TypeError('Unknown kernel type', kerneltype)

            return r

        if not t:
            t = (1./(2.*G.lmax) * np.power(2, np.arange(Nf-2, -1, -1)))

        if len(t) != Nf - 1:
            self.logger.warning('You have specified more scales than '
                                'number of filters minus 1.')

        g = [lambda x: kernel_simple_tf(t[0] * x, 'sf')]

        for i in range(Nf - 1):
            g.append(lambda x, ind=i: kernel_simple_tf(t[ind] * x, 'wavelet'))

        self.g = g
