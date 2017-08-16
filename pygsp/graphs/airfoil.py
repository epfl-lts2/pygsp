# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix

from . import Graph
from ..utils import loadmat


class Airfoil(Graph):
    r"""
    Create the airfoil graph.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Airfoil()

    """

    def __init__(self, **kwargs):

        data = loadmat('pointclouds/airfoil')
        coords = np.concatenate((data['x'], data['y']), axis=1)

        i_inds = np.reshape(data['i_inds'] - 1, 12289)
        j_inds = np.reshape(data['j_inds'] - 1, 12289)
        A = coo_matrix((np.ones(12289), (i_inds, j_inds)), shape=(4253, 4253))
        W = (A + A.T) / 2.

        plotting = {"vertex_size": 30,
                    "limits": np.array([-1e-4, 1.01*data['x'].max(),
                                        -1e-4, 1.01*data['y'].max()])}

        super(Airfoil, self).__init__(W=W, coords=coords, plotting=plotting,
                                      gtype='Airfoil', **kwargs)
