# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Heat(Filter):
    r"""Design a filter bank of heat kernels.

    The filter is defined in the spectral domain as

    .. math::
        \hat{g}(x) = \exp \left( \frac{-\tau x}{\lambda_{\text{max}}} \right),

    and is as such a low-pass filter. An application of this filter to a signal
    simulates heat diffusion.

    Parameters
    ----------
    G : graph
    tau : int or list of ints
        Scaling parameter. If a list, creates a filter bank with one filter per
        value of tau.
    normalize : bool
        Normalizes the kernel. Needs the eigenvalues.

    Examples
    --------

    Regular heat kernel.

    >>> G = graphs.Logo()
    >>> g = filters.Heat(G, tau=[5, 10])
    >>> print('{} filters'.format(g.Nf))
    2 filters
    >>> y = g.evaluate(G.e)
    >>> print('{:.2f}'.format(np.linalg.norm(y[0])))
    9.76

    Normalized heat kernel.

    >>> g = filters.Heat(G, tau=[5, 10], normalize=True)
    >>> y = g.evaluate(G.e)
    >>> print('{:.2f}'.format(np.linalg.norm(y[0])))
    1.00

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Heat(G, tau=[5, 10, 100])
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, tau=10, normalize=False, **kwargs):

        try:
            iter(tau)
        except TypeError:
            tau = [tau]

        def kernel(x, t):
            return np.exp(-t * x / G.lmax)

        g = []
        for t in tau:
            norm = np.linalg.norm(kernel(G.e, t)) if normalize else 1
            g.append(lambda x, t=t, norm=norm: kernel(x, t) / norm)

        super(Heat, self).__init__(G, g, **kwargs)
