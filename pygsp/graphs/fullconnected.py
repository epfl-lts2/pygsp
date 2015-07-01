# -*- coding: utf-8 -*-

from . import Graph
import numpy as np


class FullConnected(Graph):
    r"""
    Create a fully connected graph.

    Parameters
    ----------
    N : int
        Number of vertices (default = 10)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.FullConnected(N=5)

    """

    def __init__(self, N=10):

        tmp = np.arange(0, N).reshape(N, 1)

        self.coords = np.concatenate((np.cos(tmp*2*np.pi/N),
                                      np.sin(tmp*2*np.pi/N)),
                                     axis=1)
        self.W = np.ones((N, N))-np.identity(N)
        self.N = N
        self.gtype = "full"
        self.plotting = {"limits": np.array([-1, 1, -1, 1])}

        super(FullConnected, self).__init__(W=self.W, plotting=self.plotting,
                                            N=self.N, coords=self.coords,
                                            gtype=self.gtype)
