# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Comet(Graph):
    r"""Comet graph.

    The comet graph is a path graph with a star of degree *k* at its end.

    Parameters
    ----------
    N : int
        Number of nodes.
    k : int
        Degree of center vertex.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Comet(15, 10)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N=32, k=12, **kwargs):

        # Create weighted adjacency matrix
        i_inds = np.concatenate((np.zeros((k)), np.arange(k) + 1,
                                 np.arange(k, N - 1),
                                 np.arange(k + 1, N)))
        j_inds = np.concatenate((np.arange(k) + 1, np.zeros((k)),
                                 np.arange(k + 1, N),
                                 np.arange(k, N - 1)))

        W = sparse.csc_matrix((np.ones(np.size(i_inds)), (i_inds, j_inds)),
                              shape=(N, N))

        tmpcoords = np.zeros((N, 2))
        inds = np.arange(k) + 1
        tmpcoords[1:k + 1, 0] = np.cos(inds*2*np.pi/k)
        tmpcoords[1:k + 1, 1] = np.sin(inds*2*np.pi/k)
        tmpcoords[k + 1:, 0] = np.arange(1, N - k) + 1

        self.N = N
        self.k = k
        plotting = {"limits": np.array([-2,
                                        np.max(tmpcoords[:, 0]),
                                        np.min(tmpcoords[:, 1]),
                                        np.max(tmpcoords[:, 1])])}

        super(Comet, self).__init__(W=W, coords=tmpcoords, gtype='Comet',
                                    plotting=plotting, **kwargs)
