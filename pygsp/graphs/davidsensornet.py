# -*- coding: utf-8 -*-

from . import Graph
from pygsp.pointsclouds import PointsCloud
from pygsp.utils import distanz

import numpy as np
from scipy import sparse


class DavidSensorNet(Graph):
    r"""
    Create a sensor network.

    Parameters
    ----------
    N : int
        Number of vertices (default = 64)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.DavidSensorNet(N=500)

    """

    def __init__(self, N=64):
        if N == 64:
            david64 = PointsCloud("david64")
            W = david64.W
            coords = david64.coords

        elif N == 500:
            david500 = PointsCloud("david500")
            W = david500.W
            coords = david500.coords

        else:
            coords = np.random.rand(N, 2)

            target_dist_cutoff = -0.125 * N / 436.075 + 0.2183
            T = 0.6
            s = np.sqrt(-target_dist_cutoff**2/(2*np.log(T)))
            d = distanz(coords.T)
            W = np.exp(-np.power(d, 2)/(2.*s**2))
            W[W < T] = 0
            W[np.diag_indices(N)] = 0

        plotting = {"limits": [0, 1, 0, 1]}

        super(DavidSensorNet, self).__init__(W=W, gtype='davidsensornet',
                                             coords=coords, plotting=plotting)
