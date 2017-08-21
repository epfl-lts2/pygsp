# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Minnesota(Graph):
    r"""
    Create a community graph.

    Parameters
    ----------
    connect : bool
        Change the graph to be connected. (default = True)

    References
    ----------
    See :cite:`gleich`

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Minnesota()

    """

    def __init__(self, connect=True):

        data = utils.loadmat('pointclouds/minnesota')
        self.labels = data['labels']
        A = data['A']

        plotting = {"limits": np.array([-98, -89, 43, 50]),
                    "vertex_size": 30}

        if connect:
            # Edit adjacency matrix
            A = (A > 0).astype(int)

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

            gtype = 'minnesota'

        else:
            gtype = 'minnesota-disconnected'

        super(Minnesota, self).__init__(W=A, coords=data['xy'],
                                        gtype=gtype, plotting=plotting)
