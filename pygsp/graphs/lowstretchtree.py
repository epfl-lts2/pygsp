# -*- coding: utf-8 -*-

from . import Graph

import numpy as np
from scipy import sparse


class LowStretchTree(Graph):
    r"""
    Create a low stretch tree graph.

    Parameters
    ----------
    k : int
        2^k points on each side of the grid of vertices (default 6)

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.LowStretchTree(k=3)
    >>> G.plot()
    """

    def __init__(self, k=6, **kwargs):

        XCoords = np.array([1, 2, 1, 2])
        YCoords = np.array([1, 1, 2, 2])

        ii = np.array([0, 0, 1, 2, 2, 3])
        jj = np.array([1, 2, 1, 3, 0, 2])

        for p in range(1, k):
            ii = np.concatenate((ii, ii + 4**p, ii + 2*4**p,
                                 ii + 3*4**p, [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1],
                                 [5/3.*4**p + 1/3. - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1], [3*4**p]))
            jj = np.concatenate((jj, jj + 4**p, jj + 2*4**p, jj + 3*4**p,
                                 [5./3*4**p + 1/3. - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1],
                                 [3*4**p], [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1)+2)/3. - 1]))

            YCoords = np.kron(np.ones((2)), YCoords)
            YCoords = np.concatenate((YCoords, YCoords + 2**p))

            XCoords = np.concatenate((XCoords, XCoords + 2**p))
            XCoords = np.kron(np.ones((2)), XCoords)

        W = sparse.csc_matrix((np.ones((np.shape(ii))), (ii, jj)))
        coords = np.concatenate((XCoords[:, np.newaxis],
                                 YCoords[:, np.newaxis]),
                                axis=1)

        self.root = 4**(k - 1)

        plotting = {"edges_width": 1.25,
                    "vertex_sizee": 75,
                    "limits": np.array([0, 2**k + 1, 0, 2**k + 1])}

        super(LowStretchTree, self).__init__(W=W, coords=coords, plotting=plotting,
                                             gtype="low strech tree", **kwargs)
