# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Torus(Graph):
    r"""Sampled torus manifold.

    Parameters
    ----------
    Nv : int
        Number of vertices along the first dimension (default is 16)
    Mv : int
        Number of vertices along the second dimension (default is Nv)

    References
    ----------
    See :cite:`strang1999discrete` for more informations.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Torus(10)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> G.plot(ax=ax2)
    >>> _ = ax2.set_zlim(-1.5, 1.5)

    """

    def __init__(self, Nv=16, Mv=None, **kwargs):

        if Mv is None:
            Mv = Nv

        # Create weighted adjancency matrix
        K = 2 * Nv
        J = 2 * Mv
        i_inds = np.zeros((K*Mv + J*Nv), dtype=float)
        j_inds = np.zeros((K*Mv + J*Nv), dtype=float)

        tmpK = np.arange(K, dtype=int)
        tmpNv1 = np.arange(Nv - 1)
        tmpNv = np.arange(Nv)

        for i in range(Mv):
            i_inds[i*K + tmpK] = i*Nv + \
                np.concatenate((np.array([Nv - 1]), tmpNv1, tmpNv))

            j_inds[i*K + tmpK] = i*Nv + \
                np.concatenate((tmpNv, np.array([Nv - 1]), tmpNv1))

        tmp2Nv = np.arange(2*Nv, dtype=int)

        for i in range(Mv - 1):
            i_inds[K*Mv + i*2*Nv + tmp2Nv] = \
                np.concatenate((i*Nv + tmpNv, (i + 1)*Nv + tmpNv))

            j_inds[K*Mv + i*2*Nv + tmp2Nv] = \
                np.concatenate(((i + 1)*Nv + tmpNv, i*Nv + tmpNv))

        i_inds[K*Mv + (Mv - 1)*2*Nv + tmp2Nv] = \
            np.concatenate((tmpNv, (Mv - 1)*Nv + tmpNv))

        j_inds[K*Mv + (Mv - 1)*2*Nv + tmp2Nv] = \
            np.concatenate(((Mv - 1)*Nv + tmpNv, tmpNv))

        W = sparse.csc_matrix((np.ones((K*Mv + J*Nv)), (i_inds, j_inds)),
                              shape=(Mv*Nv, Mv*Nv))

        # Create coordinate
        T = 1.5 + np.sin(np.arange(Mv)*2*np.pi/Mv).reshape(1, Mv)
        U = np.cos(np.arange(Mv)*2*np.pi/Mv).reshape(1, Mv)
        xtmp = np.cos(np.arange(Nv).reshape(Nv, 1)*2*np.pi/Nv)*T
        ytmp = np.sin(np.arange(Nv).reshape(Nv, 1)*2*np.pi/Nv)*T
        ztmp = np.kron(np.ones((Nv, 1)), U)
        coords = np.concatenate((np.reshape(xtmp, (Mv*Nv, 1), order='F'),
                                 np.reshape(ytmp, (Mv*Nv, 1), order='F'),
                                 np.reshape(ztmp, (Mv*Nv, 1), order='F')),
                                axis=1)
        self.Nv = Nv
        self.Mv = Nv

        plotting = {
            'vertex_size': 60,
            'limits': np.array([-2.5, 2.5, -2.5, 2.5, -2.5, 2.5])
        }

        super(Torus, self).__init__(W=W, gtype='Torus', coords=coords,
                                    plotting=plotting, **kwargs)
