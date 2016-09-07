# -*- coding: utf-8 -*-

from . import Graph

import numpy as np
from scipy import sparse


class RandomRing(Graph):
    r"""
    Create a ring graph.

    Parameters
    ----------
    N : int
        Number of vertices (default = 64)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.RandomRing(N=16)

    """

    def __init__(self, N=64):

        position = np.sort(np.random.rand(N), axis=0)

        weight = N*np.diff(position)
        weightend = N*(1 + position[0] - position[-1])

        inds_j = np.arange(1, N)
        inds_i = np.arange(N - 1)

        W = sparse.csc_matrix((weight, (inds_i, inds_j)), shape=(N, N))
        W = W.tolil()
        W[N - 1, 0] = weightend
        W = W + W.T

        plotting = {'limits': np.array([-1, 1, -1, 1])}

        super(RandomRing, self).__init__(W=W, gtype='random-ring', plotting=plotting)
