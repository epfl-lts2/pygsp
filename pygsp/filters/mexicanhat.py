# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


class MexicanHat(Filter):
    r"""Design a filter bank of Mexican hat wavelets.

    The Mexican hat wavelet is the second oder derivative of a Gaussian. Since
    we express the filter in the Fourier domain, we find:

    .. math:: \hat{g}_b(x) = x * \exp(-x)

    for the band-pass filter. Note that in our convention the eigenvalues of
    the Laplacian are equivalent to the square of graph frequencies,
    i.e. :math:`x = \lambda^2`.

    The low-pass filter is given by

    .. math: \hat{g}_l(x) = \exp(-x^4).

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters to cover the interval [0, lmax].
    lpfactor : int
        Low-pass factor. lmin=lmax/lpfactor will be used to determine scales.
        The scaling function will be created to fill the low-pass gap.
    scales : array-like
        Scales to be used.
        By default, initialized with :func:`pygsp.utils.compute_log_scales`.
    normalize : bool
        Whether to normalize the wavelet by the factor ``sqrt(scales)``.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.MexicanHat(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, lpfactor=20, scales=None, normalize=False,
                 **kwargs):

        lmin = G.lmax / lpfactor

        if scales is None:
            scales = utils.compute_log_scales(lmin, G.lmax, Nf-1)

        if len(scales) != Nf - 1:
            raise ValueError('len(scales) should be Nf-1.')

        def band_pass(x):
            return x * np.exp(-x)

        def low_pass(x):
            return np.exp(-x**4)

        kernels = [lambda x: 1.2 * np.exp(-1) * low_pass(x / 0.4 / lmin)]

        for i in range(Nf - 1):

            def kernel(x, i=i):
                norm = np.sqrt(scales[i]) if normalize else 1
                return norm * band_pass(scales[i] * x)

            kernels.append(kernel)

        super(MexicanHat, self).__init__(G, kernels, **kwargs)
