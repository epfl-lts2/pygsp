# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


class SimpleTight(Filter):
    r"""Design a simple tight frame filter bank (tight frame).

    These filters have been designed to be a simple tight frame wavelet filter
    bank. The kernel is similar to Meyer, but simpler. The function is
    essentially :math:`\sin^2(x)` in ascending part and :math:`\cos^2` in
    descending part.

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters to cover the interval [0, lmax].
    scales : array_like
        Scales to be used. Defaults to log scale.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.SimpleTight(G)
    >>> g
    SimpleTight(in=1, out=6)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, scales=None):

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

            def h(x):
                return np.sin(np.pi*x/2.)**2

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            r = np.zeros(x.shape)
            if kerneltype == 'sf':
                r[r1ind] = 1.
                r[r2ind] = np.sqrt(1 - h(4*x[r2ind] - 1)**2)
            elif kerneltype == 'wavelet':
                r[r2ind] = h(4*(x[r2ind] - 1/4.))
                r[r3ind] = np.sqrt(1 - h(2*x[r3ind] - 1)**2)
            else:
                raise TypeError('Unknown kernel type', kerneltype)

            return r

        if not scales:
            scales = (1./(2.*G.lmax) * np.power(2, np.arange(Nf-2, -1, -1)))
        self.scales = scales

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        kernels = [lambda x: kernel(scales[0] * x, 'sf')]

        for i in range(Nf - 1):
            kernels.append(lambda x, i=i: kernel(scales[i] * x, 'wavelet'))

        super(SimpleTight, self).__init__(G, kernels)
