# -*- coding: utf-8 -*-

from . import Graph

import numpy as np
from scipy import sparse


class Path(Graph):
    r"""
    Create a path graph.

    Parameters
    ----------
    N : int
        Number of vertices (default = 32)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Path(N=16)

    See :cite:`strang1999discrete` for more informations.
    """

    def __init__(self, N=16):

        inds_i = np.concatenate((np.arange(N - 1), np.arange(1, N)))
        inds_j = np.concatenate((np.arange(1, N), np.arange(N - 1)))

        self.W = sparse.csc_matrix((np.ones((2*(N - 1))), (inds_i, inds_j)),
                                   shape=(N, N))
        self.coords = np.concatenate((np.expand_dims(np.arange(N) + 1, axis=1),
                                      np.zeros((N, 1))),
                                     axis=1)
        self.plotting = {"limits": np.array([0, N + 1, -1, 1])}
        self.gtype = "path"
        self.N = N

        super(Path, self).__init__(W=self.W, coords=self.coords,
                                   plotting=self.plotting, gtype=self.gtype)
