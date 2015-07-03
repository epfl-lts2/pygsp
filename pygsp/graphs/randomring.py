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

        self.W = W + W.getH()

        self.coords = np.concatenate((np.expand_dims(np.cos(position*2*np.pi),
                                      axis=1),
                                      np.expand_dims(np.sin(position*2*np.pi),
                                      axis=1)),
                                     axis=1)

        self.N = N
        self.limits = np.array([-1, 1, -1, 1])
        self.gtype = 'random-ring'

        super(RandomRing, self).__init__(N=self.N, W=self.W,
                                         gtype=self.gtype,
                                         coords=self.coords,
                                         limits=self.limits)
