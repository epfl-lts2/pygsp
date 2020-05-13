# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Comet(Graph):
    r"""Comet graph.

    The comet is a path graph with a star of degree `k` at one end.
    Equivalently, the comet is a star made of `k` branches, where a branch of
    length `N-k` acts as the tail.
    The central vertex has degree `N-1`, the others have degree 1.

    Parameters
    ----------
    N : int
        Number of vertices.
    k : int
        Degree of central vertex.

    See Also
    --------
    Path : Comet without star
    Star : Comet without tail (path)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Comet(15, 10)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, N=32, k=12, **kwargs):

        if k > N-1:
            raise ValueError('The degree of the central vertex k={} must be '
                             'smaller than the number of vertices N={}.'
                             ''.format(k, N))

        self.k = k

        sources = np.concatenate((
            np.zeros(k), np.arange(k)+1,  # star
            np.arange(k, N-1), np.arange(k+1, N)  # tail (path)
        ))
        targets = np.concatenate((
            np.arange(k)+1, np.zeros(k),  # star
            np.arange(k+1, N), np.arange(k, N-1)  # tail (path)
        ))
        n_edges = N - 1
        weights = np.ones(2*n_edges)
        W = sparse.csr_matrix((weights, (sources, targets)), shape=(N, N))

        indices = np.arange(k) + 1
        coords = np.zeros((N, 2))
        coords[1:k+1, 0] = np.cos(indices*2*np.pi/k)
        coords[1:k+1, 1] = np.sin(indices*2*np.pi/k)
        coords[k+1:, 0] = np.arange(1, N-k) + 1

        super(Comet, self).__init__(W, coords=coords, **kwargs)

    def _get_extra_repr(self):
        return dict(k=self.k)
