# -*- coding: utf-8 -*-

from . import Graph
from pygsp.pointsclouds import PointsCloud

import numpy as np
from scipy import sparse


class Airfoil(Graph):
    r"""
    Create the airfoil graph.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Airfoil()

    """

    def __init__(self, **kwargs):

        airfoil = PointsCloud("airfoil")
        i_inds = airfoil.i_inds
        j_inds = airfoil.j_inds

        A = sparse.coo_matrix((np.ones((12289)),
                              (np.reshape(i_inds - 1, (12289)),
                               np.reshape(j_inds - 1, (12289)))),
                              shape=(4253, 4253))
        W = (A + A.T) / 2.

        plotting = {"vertex_size": 30,
                    "limits": np.array([-1e-4, 1.01*np.max(airfoil.x),
                                        -1e-4, 1.01*np.max(airfoil.y)])}

        super(Airfoil, self).__init__(W=W, coords=airfoil.coords,
                                      plotting=plotting, gtype='Airfoil',
                                      **kwargs)
