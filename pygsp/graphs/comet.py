# -*- coding: utf-8 -*-

from . import Graph

import numpy as np
from scipy import sparse


class Comet(Graph):
    r"""
    Create a Comet graph.

    Parameters
    ----------
    Nv : int
        Number of vertices along the first dimension (default is 16)
    Mv : int
        Number of vertices along the second dimension (default is Nv)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Comet() # (== graphs.Comet(Nv=32, k=12))

    """

    def __init__(self, Nv=32, k=12, **kwargs):

        # Create weighted adjancency matrix
        i_inds = np.concatenate((np.zeros((k)), np.arange(k) + 1,
                                 np.arange(k, Nv - 1),
                                 np.arange(k + 1, Nv)))
        j_inds = np.concatenate((np.arange(k) + 1, np.zeros((k)),
                                 np.arange(k + 1, Nv),
                                 np.arange(k, Nv - 1)))

        W = sparse.csc_matrix((np.ones(np.size(i_inds)), (i_inds, j_inds)),
                              shape=(Nv, Nv))

        tmpcoords = np.zeros((Nv, 2))
        inds = np.arange(k) + 1
        tmpcoords[1:k + 1, 0] = np.cos(inds*2*np.pi/k)
        tmpcoords[1:k + 1, 1] = np.sin(inds*2*np.pi/k)
        tmpcoords[k + 1:, 0] = np.arange(1, Nv - k) + 1

        self.Nv = Nv
        self.k = k
        plotting = {"limits": np.array([-2,
                                        np.max(tmpcoords[:, 0]),
                                        np.min(tmpcoords[:, 1]),
                                        np.max(tmpcoords[:, 1])])}

        super(Comet, self).__init__(W=W, coords=tmpcoords, gtype='Comet',
                                    plotting=plotting, **kwargs)
