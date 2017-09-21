# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Ring(Graph):
    r"""K-regular ring graph.

    Parameters
    ----------
    N : int
        Number of vertices.
    k : int
        Number of neighbors in each direction.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=10)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N=64, k=1, **kwargs):

        if 2*k > N:
            raise ValueError('Too many neighbors requested.')

        if 2*k == N:
            num_edges = N * (k - 1) + k
        else:
            num_edges = N * k

        i_inds = np.zeros((2 * num_edges))
        j_inds = np.zeros((2 * num_edges))

        tmpN = np.arange(N, dtype=int)
        for i in range(min(k, (N - 1) // 2)):
            i_inds[2*i * N + tmpN] = tmpN
            j_inds[2*i * N + tmpN] = np.remainder(tmpN + i + 1, N)
            i_inds[(2*i + 1)*N + tmpN] = np.remainder(tmpN + i + 1, N)
            j_inds[(2*i + 1)*N + tmpN] = tmpN

        if 2*k == N:
            i_inds[2*N*(k - 1) + tmpN] = tmpN
            i_inds[2*N*(k - 1) + tmpN] = np.remainder(tmpN + k + 1, N)

        W = sparse.csc_matrix((np.ones((2*num_edges)), (i_inds, j_inds)),
                              shape=(N, N))

        plotting = {'limits': np.array([-1, 1, -1, 1])}

        gtype = 'ring' if k == 1 else 'k-ring'
        self.k = k

        super(Ring, self).__init__(W=W, gtype=gtype, plotting=plotting,
                                   **kwargs)

        self.set_coordinates('ring2D')
