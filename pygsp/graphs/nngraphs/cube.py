# -*- coding: utf-8 -*-

from . import NNGraph

import numpy as np
from math import floor


class Cube(NNGraph):
    r"""
    Creates the graph of an hyper-cube.

    Parameters
    ----------
    radius : float
        Edge lenght (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')

    Examples
    --------
    >>> from pygsp import graphs
    >>> radius = 5
    >>> G = graphs.Cube(radius=radius)

    """

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling="random", **kwargs):
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.nb_dim > 3:
            raise NotImplementedError("Dimension > 3 not supported yet !")

        if self.sampling == "random":
            if self.nb_dim == 2:
                pts = np.random.rand(self.nb_pts, self.nb_pts)

            elif self.nb_dim == 3:
                n = floor(self.nb_pts/6.)

                pts = np.zeros((n*6, 3))
                pts[:n, 1:] = np.random.rand(n, 2)
                pts[n:2*n, :] = np.concatenate((np.ones((n, 1)),
                                                np.random.rand(n, 2)),
                                               axis=1)

                pts[2*n:3*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.zeros((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)
                pts[3*n:4*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.ones((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)

                pts[4*n:5*n, :2] = np.random.rand(n, 2)
                pts[5*n:6*n, :] = np.concatenate((np.random.rand(n, 2),
                                                  np.ones((n, 1))),
                                                 axis=1)

        else:
            raise ValueError("Unknown sampling !")

        super(Cube, self).__init__(Xin=pts, k=10, gtype="Cube", **kwargs)
