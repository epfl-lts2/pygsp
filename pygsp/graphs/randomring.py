# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class RandomRing(Graph):
    r"""Ring graph with randomly sampled nodes.

    Parameters
    ----------
    N : int
        Number of vertices.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.RandomRing(N=10, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])
    >>> _ = axes[1].set_xlim(-1.1, 1.1)
    >>> _ = axes[1].set_ylim(-1.1, 1.1)

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
        W[0, N-1] = weight_end
        W = utils.symmetrize(W, method='triu')

        angle = position * 2 * np.pi
        coords = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        plotting = {'limits': np.array([-1, 1, -1, 1])}

        super(RandomRing, self).__init__(W=W, gtype='random-ring',
                                         coords=coords, plotting=plotting,
                                         **kwargs)
