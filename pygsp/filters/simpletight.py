# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class SimpleTight(Filter):
    r"""
    Design a simple tight frame filter bank.

    These filters have been designed to be a simple tight frame wavelet filter
    bank. The kernel is similar to Meyer, but simpler. The function is
    essentially :math:`\sin^2(x)` in ascending part and :math:`\cos^2` in
    descending part.

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters to cover the interval [0, lmax].
    scales : array-like
        Scales to be used. Defaults to log scale.

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> g = filters.SimpleTight(G)

    """

    def __init__(self, G, Nf=6, scales=None, **kwargs):

        def kernel(x, kerneltype):
            r"""
            Evaluates 'simple' tight-frame kernel.

            * simple tf wavelet kernel: supported on [1/4, 1]
            * simple tf scaling function kernel: supported on [0, 1/2]

            Parameters
            ----------
            x : ndarray
                Array of independent variable values
            kerneltype : str
                Can be either 'sf' or 'wavelet'

            Returns
            -------
            r : ndarray

            """

            l1 = 0.25
            l2 = 0.5
            l3 = 1.

            h = lambda x: np.sin(np.pi*x/2.)**2

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

        if not scales:
            scales = (1./(2.*G.lmax) * np.power(2, np.arange(Nf-2, -1, -1)))

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        g = [lambda x: kernel(scales[0] * x, 'sf')]

        for i in range(Nf - 1):
            g.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        super(SimpleTight, self).__init__(G, g, **kwargs)
