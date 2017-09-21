# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Airfoil(Graph):
    r"""Airfoil graph.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Airfoil()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(show_edges=True, ax=axes[1])

    """

    def __init__(self, **kwargs):

        data = utils.loadmat('pointclouds/airfoil')
        coords = np.concatenate((data['x'], data['y']), axis=1)

        i_inds = np.reshape(data['i_inds'] - 1, 12289)
        j_inds = np.reshape(data['j_inds'] - 1, 12289)
        A = sparse.coo_matrix((np.ones(12289), (i_inds, j_inds)), shape=(4253, 4253))
        W = (A + A.T) / 2.

        plotting = {"vertex_size": 30,
                    "limits": np.array([-1e-4, 1.01*data['x'].max(),
                                        -1e-4, 1.01*data['y'].max()])}

        super(Airfoil, self).__init__(W=W, coords=coords, plotting=plotting,
                                      gtype='Airfoil', **kwargs)
