# -*- coding: utf-8 -*-

from . import Filter

import numpy as np


class Gabor(Filter):
    """
    Gabor Filterbank

    Inherits its methods from Filters
    Parameters
    ----------
    G : Graph
        Graph structur
    k : lambda function
        kernel

    Returns
    -------
    g : Gabor

    Note
    ----
    This function create a filterbank with the kernel *k*. Every filter is
    centered in a different frequency

    Examples
    --------
    >>> from pygsp import grpahs, filters
    >>> G = graphs.Logo()
    >>> k = lambda x: x/(1.-x)
    >>> g = filters.Gabor(G, k);

    Author: Nathanael Perraudin
    Date  : 13 June 2014
    """
    def __init__(self, G, k, **kwargs):
        super(Gabor, self).__init__(G, **kwargs)

        if not hasattr(G, 'e'):
            self.logger.info('Filter Gabor will calculate and set'
                             ' the eigenvalues to normalize the kernel')
            G.compute_fourier_basis()

        Nf = np.shape(G.e)[0]

        g = []
        for i in range(Nf):
            g.append(lambda x, ii=i: k(x - G.e[ii]))

        return g
