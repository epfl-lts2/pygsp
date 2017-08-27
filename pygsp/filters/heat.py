# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Heat(Filter):
    r"""
    Design an heat low-pass filter (simulates heat diffusion when applied).

    The filter is defined in the spectral domain as

    .. math::
        \hat{g}(x) = \exp \left( \frac{-\tau x}{\lambda_{\text{max}}} \right).

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
    >>> import numpy as np
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()

    Regular heat kernel.

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
