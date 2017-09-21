# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Grid2d(Graph):
    r"""2-dimensional grid graph.

    Parameters
    ----------
    N1 : int
        Number of vertices along the first dimension.
    N2 : int
        Number of vertices along the second dimension (default N1).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Grid2d(N1=5, N2=4)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N1=16, N2=None, **kwargs):

        if N2 is None:
            N2 = N1

        N = N1 * N2

        # Filling up the weight matrix this way is faster than
        # looping through all the grid points:
        diag_1 = np.ones(N - 1)
        diag_1[(N2 - 1)::N2] = 0
        diag_2 = np.ones(N - N2)
        W = sparse.diags(diagonals=[diag_1, diag_2],
                         offsets=[-1, -N2],
                         shape=(N, N),
                         format='csr',
                         dtype='float')
        W = utils.symmetrize(W, method='tril')

        x = np.kron(np.ones((N1, 1)), (np.arange(N2)/float(N2)).reshape(N2, 1))
        y = np.kron(np.ones((N2, 1)), np.arange(N1)/float(N1)).reshape(N, 1)
        y = np.sort(y, axis=0)[::-1]
        coords = np.concatenate((x, y), axis=1)

        plotting = {"limits": np.array([-1. / N2, 1 + 1. / N2,
                                        1. / N1, 1 + 1. / N1])}

        super(Grid2d, self).__init__(W=W, gtype='2d-grid', coords=coords,
                                     plotting=plotting, **kwargs)
