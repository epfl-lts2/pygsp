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
        K = 2*(Nv - 1)
        J = 2*(Mv - 1)

        i_inds = np.zeros((K*Mv + J*Nv), dtype=float)
        j_inds = np.zeros((K*Mv + J*Nv), dtype=float)

        tmpK = np.arange(K, dtype=int)
        tmpNv1 = np.arange(Nv - 1)
        for i in range(Mv):
            i_inds[i*K + tmpK] = i*Nv + \
                np.concatenate((tmpNv1, tmpNv1 + 1))
            j_inds[i*K + tmpK] = i*Nv + \
                np.concatenate((tmpNv1 + 1, tmpNv1))

        tmp2Nv = np.arange(2*Nv, dtype=int)
        tmpNv = np.arange(Nv)
        for i in range(Mv-1):
            i_inds[(K*Mv) + i*2*Nv + tmp2Nv] = \
                np.concatenate((i*Nv + tmpNv, (i + 1)*Nv + tmpNv))

            j_inds[(K*Mv) + i*2*Nv + tmp2Nv] = \
                np.concatenate(((i + 1)*Nv + tmpNv, i*Nv + tmpNv))

        W = sparse.csc_matrix((np.ones((K*Mv + J*Nv)), (i_inds, j_inds)),
                              shape=(Mv*Nv, Mv*Nv))

        xtmp = np.kron(np.ones((Mv, 1)), (np.arange(Nv)/float(Nv)).reshape(Nv,
                                                                           1))
        ytmp = np.sort(np.kron(np.ones((Nv, 1)),
                               np.arange(Mv)/float(Mv)).reshape(Mv*Nv, 1),
                       axis=0)

        coords = np.concatenate((xtmp, ytmp), axis=1)

        self.Nv = Nv
        self.Mv = Mv
        plotting = {"vertex_size": 30,
                    "limits": np.array([-1./self.Nv, 1 + 1./self.Nv,
                                        1./self.Mv, 1 + 1./self.Mv])}

        super(Grid2d, self).__init__(W=W, gtype='2d-grid', coords=coords,
                                     plotting=plotting, **kwargs)
