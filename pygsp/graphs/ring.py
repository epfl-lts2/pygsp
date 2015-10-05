# -*- coding: utf-8 -*-

from . import Graph

import numpy as np
from scipy import sparse
from math import floor


class Ring(Graph):
    r"""
    Create a ring graph.

    Parameters
    ----------
    N : int
        Number of vertices (default is 64)
    k : int
        Number of neighbors in each directions (default is 1)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Ring()

    """

    def __init__(self, N=64, k=1, **kwargs):

        if k > N/2.:
            raise ValueError("Too many neighbors requested.")

        # Create weighted adjancency matrix
        if k == N/2.:
            num_edges = N*(k - 1) + N/2.
        else:
            num_edges = N*k

        i_inds = np.zeros((2*num_edges))
        j_inds = np.zeros((2*num_edges))

        tmpN = np.arange(N, dtype=int)
        for i in range(min(k, floor((N - 1)/2.))):
            i_inds[i*2*N + tmpN] = tmpN
            j_inds[i*2*N + tmpN] = np.remainder(tmpN + i + 1, N)
            i_inds[(i*2 + 1)*N + tmpN] = np.remainder(tmpN + i + 1, N)
            j_inds[(i*2 + 1)*N + tmpN] = tmpN

        if k == N/2.:
            i_inds[2*N*(k - 1) + tmpN] = tmpN
            i_inds[2*N*(k - 1) + tmpN] = np.remainder(tmpN + k + 1, N)

        self.W = sparse.csc_matrix((np.ones((2*num_edges)), (i_inds, j_inds)),
                                   shape=(N, N))

        self.coords = np.concatenate((np.cos(np.arange(N).reshape(N, 1)
                                             * 2 * np.pi/float(N)),
                                      np.sin(np.arange(N).reshape(N, 1)
                                             * 2 * np.pi/float(N))),
                                     axis=1)

        self.plotting = {"limits": np.array([-1, 1, -1, 1])}

        if k == 1:
            self.gtype = "ring"
        else:
            self.gtype = "k-ring"

        self.N = N
        self.k = k

        super(Ring, self).__init__(W=self.W, N=self.N, gtype=self.gtype,
                                   coords=self.coords, plotting=self.plotting,
                                   **kwargs)
