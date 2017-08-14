# -*- coding: utf-8 -*-

import numpy as np

from . import Filter


class Gabor(Filter):
    r"""
    Gabor filterbank

    Parameters
    ----------
    G : graph
    k : lambda function
        kernel

    Notes
    -----
    This function create a filterbank with the kernel *k*. Every filter is
    centered in a different frequency.

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> k = lambda x: x/(1.-x)
    >>> F = filters.Gabor(G, k);

    """
    def __init__(self, G, k, **kwargs):
        super(Gabor, self).__init__(G, **kwargs)

        if not hasattr(G, 'e'):
            self.logger.info('Filter Gabor will calculate and set'
                             ' the eigenvalues to normalize the kernel')
            G.compute_fourier_basis()

        Nf = np.shape(G.e)[0]

        self.g = []
        for i in range(Nf):
            self.g.append(lambda x, ii=i: k(x - G.e[ii]))
