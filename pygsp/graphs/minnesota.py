# -*- coding: utf-8 -*-

from pygsp.pointsclouds import PointsCloud
from . import Graph

import numpy as np


class Minnesota(Graph):
    r"""
    Create a community graph.

    Parameters
    ----------
    connect : bool
        Change the graph to be connected. (default = True)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Minnesota()

    References
    ----------
    See :cite:`gleich`

    """

    def __init__(self, connect=True):
        minnesota = PointsCloud('minnesota')

        self.N = np.shape(minnesota.A)[0]
        self.coords = minnesota.coords
        self.plotting = {"limits": np.array([-98, -89, 43, 50]),
                         "vertex_size": 30}

        if connect:
            # Edit adjacency matrix
            A = minnesota.A.tolil()

            # clean minnesota graph
            A.setdiag(0)

            # missing edge needed to connect graph
            A[349, 355] = 1
            A[355, 349] = 1

            # change a handful of 2 values back to 1
            A[86, 88] = 1
            A[86, 88] = 1
            A[345, 346] = 1
            A[346, 345] = 1
            A[1707, 1709] = 1
            A[1709, 1707] = 1
            A[2289, 2290] = 1
            A[2290, 2289] = 1

            self.W = A
            self.gtype = 'minnesota'

        else:
            self.W = A
            self.gtype = 'minnesota-disconnected'

        super(Minnesota, self).__init__(W=self.W, gtype=self.gtype,
                                        coords=self.coords, N=self.N,
                                        plotting=self.plotting)
