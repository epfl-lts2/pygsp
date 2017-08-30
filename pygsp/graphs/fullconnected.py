# -*- coding: utf-8 -*-

import numpy as np

from . import Graph  # prevent circular import in Python < 3.5


class FullConnected(Graph):
    r"""Fully connected graph.

    Parameters
    ----------
    N : int
        Number of vertices (default = 10)

    Examples
    --------
    >>> G = graphs.FullConnected()

    """

    def __init__(self, N=10):

        W = np.ones((N, N)) - np.identity(N)
        plotting = {'limits': np.array([-1, 1, -1, 1])}

        super(FullConnected, self).__init__(W=W, gtype='full',
                                            plotting=plotting)
