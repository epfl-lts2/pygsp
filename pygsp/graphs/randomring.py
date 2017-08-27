# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class RandomRing(Graph):
    r"""
    Create a ring graph.

    Parameters
    ----------
    N : int
        Number of vertices.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.RandomRing(N=16)

    """

    def __init__(self, N=64, seed=None, **kwargs):

        rs = np.random.RandomState(seed)
        position = np.sort(rs.uniform(size=N), axis=0)

        weight = N * np.diff(position)
        weight_end = N * (1 + position[0] - position[-1])

        inds_i = np.arange(0, N-1)
        inds_j = np.arange(1, N)

        W = sparse.csc_matrix((weight, (inds_i, inds_j)), shape=(N, N))
        W = W.tolil()
        W[N-1, 0] = weight_end
        W = W + W.T

        angle = position * 2 * np.pi
        coords = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        plotting = {'limits': np.array([-1, 1, -1, 1])}

        super(RandomRing, self).__init__(W=W, gtype='random-ring',
                                         coords=coords, plotting=plotting,
                                         **kwargs)
