# -*- coding: utf-8 -*-

from . import Graph
from pygsp.pointsclouds import PointsCloud
from pygsp.utils import distanz

import numpy as np
from scipy import sparse
from math import sqrt, log


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
        self.N = N

        if self.N == 64:
            david64 = PointsCloud("david64")
            self.W = david64.W
            self.N = david64.N
            self.coords = david64.coords

        elif self.N == 500:
            david500 = PointsCloud("david500")
            self.W = david500.W
            self.N = david500.N
            self.coords = david500.coords

        else:
            self.coords = np.random.rand(self.N, 2)

            target_dist_cutoff = -0.125*self.N/436.075+0.2183
            T = 0.6
            s = sqrt(-target_dist_cutoff**2/(2.*log(T)))
            d = distanz(self.coords.conj().T)
            W = np.exp(-np.power(d, 2)/2.*s**2)
            W = np.where(W < T, 0, W)
            W -= np.diag(np.diag(W))
            self.W = sparse.lil_matrix(W)

        self.gtype = 'davidsensornet'
        self.plotting = {"limits": [0, 1, 0, 1]}

        super(DavidSensorNet, self).__init__(W=self.W, plotting=self.plotting,
                                             N=self.N, coords=self.coords,
                                             gtype=self.gtype)
