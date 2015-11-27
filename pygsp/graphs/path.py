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

    Reference
    ---------
    See :cite:`strang1999discrete` for more informations.

    """

    def __init__(self, N=16):

        inds_i = np.concatenate((np.arange(N - 1), np.arange(1, N)))
        inds_j = np.concatenate((np.arange(1, N), np.arange(N - 1)))

        W = sparse.csc_matrix((np.ones((2*(N - 1))), (inds_i, inds_j)),
                              shape=(N, N))
        coords = np.concatenate(((np.arange(N) + 1)[:, np.newaxis],
                                 np.zeros((N, 1))),
                                axis=1)
        plotting = {"limits": np.array([0, N + 1, -1, 1])}

        super(Path, self).__init__(W=W, coords=coords, gtype=self.gtype,
                                   plotting='path')
