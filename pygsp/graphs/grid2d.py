# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from . import Graph


class Grid2d(Graph):
    r"""
    Create a 2 dimensional grid graph.

    Parameters
    ----------
    Nv : int
        Number of vertices along the first dimension (default is 16)
    Mv : int
        Number of vertices along the second dimension (default is Nv)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Grid2d(Nv=32)

    """

    def __init__(self, Nv=16, Mv=None, **kwargs):
        if not Mv:
            Mv = Nv

        # Create weighted adjacency matrix
        K = 2*(Nv-1)
        J = 2*(Mv-1)

        i_inds = np.zeros((K*Mv + J*Nv), dtype=float)
        j_inds = np.zeros((K*Mv + J*Nv), dtype=float)

        for i in range(Mv):
            i_inds[i*K + np.arange(K)] = i*Nv + \
                np.concatenate((np.arange(Nv-1), np.arange(1, Nv)))
            j_inds[i*K + np.arange(K)] = i*Nv + \
                np.concatenate((np.arange(1, Nv), np.arange(Nv-1)))

        for i in range(Mv-1):
            i_inds[(K*Mv) + i*2*Nv + np.arange(2*Nv)] = \
                np.concatenate((i*Nv + np.arange(Nv),
                                (i+1)*Nv + np.arange(Nv)))

            j_inds[(K*Mv) + i*2*Nv + np.arange(2*Nv)] = \
                np.concatenate(((i+1)*Nv + np.arange(Nv),
                                i*Nv + np.arange(Nv)))

        self.W = sparse.csc_matrix((np.ones((K*Mv+J*Nv)), (i_inds, j_inds)),
                                   shape=(Mv*Nv, Mv*Nv))

        xtmp = np.kron(np.ones((Mv, 1)), (np.arange(Nv)/float(Nv)).reshape(Nv,
                                                                           1))
        ytmp = np.sort(np.kron(np.ones((Nv, 1)),
                               np.arange(Mv)/float(Mv)).reshape(Mv*Nv, 1),
                       axis=0)

        self.coords = np.concatenate((xtmp, ytmp), axis=1)

        self.N = Nv * Mv
        self.Nv = Nv
        self.Mv = Mv
        self.gtype = '2d-grid'
        self.plotting = {"limits": np.array([-1./self.Nv, 1 + 1./self.Nv,
                                             1./self.Mv, 1 + 1./self.Mv]),
                         "vertex_size": 30}

        super(Grid2d, self).__init__(N=self.N, W=self.W, gtype=self.gtype,
                                     plotting=self.plotting,
                                     coords=self.coords, **kwargs)
