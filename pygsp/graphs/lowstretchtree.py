# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class LowStretchTree(Graph):
    r"""Low stretch tree.

    Build the root of a low stretch tree on a grid of points. There are
    :math:`2k` points on each side of the grid, and therefore :math:`2^{2k}`
    vertices in total. The edge weights are all equal to 1.

    Parameters
    ----------
    k : int
        :math:`2^k` points on each side of the grid of vertices.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.LowStretchTree(k=2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, k=6, **kwargs):

        self.k = k

        XCoords = np.array([1, 2, 1, 2], dtype=int)
        YCoords = np.array([1, 1, 2, 2], dtype=int)

        ii = np.array([0, 0, 1, 2, 2, 3], dtype=int)
        jj = np.array([1, 2, 1, 3, 0, 2], dtype=int)

        for p in range(1, k):
            ii = np.concatenate((ii, ii + 4**p, ii + 2*4**p, ii + 3*4**p,
                                 [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1) + 2) // 3 - 1],
                                 [(5*4**p + 1) // 3 - 1],
                                 [4**p + (4**(p+1) + 2) // 3 - 1], [3*4**p]))
            jj = np.concatenate((jj, jj + 4**p, jj + 2*4**p, jj + 3*4**p,
                                 [(5*4**p + 1) // 3 - 1],
                                 [4**p + (4**(p+1) + 2) // 3 - 1],
                                 [3*4**p], [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1) + 2) // 3 - 1]))

            YCoords = np.kron(np.ones((2), dtype=int), YCoords)
            YCoords = np.concatenate((YCoords, YCoords + 2**p))

            XCoords = np.concatenate((XCoords, XCoords + 2**p))
            XCoords = np.kron(np.ones((2), dtype=int), XCoords)

        W = sparse.csc_matrix((np.ones_like(ii), (ii, jj)))
        coords = np.concatenate((XCoords[:, np.newaxis],
                                 YCoords[:, np.newaxis]),
                                axis=1)

        self.root = 4**(k - 1)

        plotting = {"edges_width": 1.25,
                    "vertex_size": 75,
                    "limits": np.array([0, 2**k + 1, 0, 2**k + 1])}

        super(LowStretchTree, self).__init__(W,
                                             coords=coords,
                                             plotting=plotting,
                                             **kwargs)

    def _get_extra_repr(self):
        return dict(k=self.k)
