# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class Cube(NNGraph):
    r"""Hyper-cube (NN-graph).

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
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Cube(seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=0.5)
    >>> G.plot(ax=ax2)

    """

    def __init__(self,
                 radius=1,
                 nb_pts=300,
                 nb_dim=3,
                 sampling='random',
                 seed=None,
                 **kwargs):

        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling
        rs = np.random.RandomState(seed)

        if self.nb_dim > 3:
            raise NotImplementedError("Dimension > 3 not supported yet !")

        if self.sampling == "random":
            if self.nb_dim == 2:
                pts = rs.rand(self.nb_pts, self.nb_dim)

            elif self.nb_dim == 3:
                n = self.nb_pts // 6

                pts = np.zeros((n*6, 3))
                pts[:n, 1:] = rs.rand(n, 2)
                pts[n:2*n, :] = np.concatenate((np.ones((n, 1)),
                                                rs.rand(n, 2)),
                                               axis=1)

                pts[2*n:3*n, :] = np.concatenate((rs.rand(n, 1),
                                                  np.zeros((n, 1)),
                                                  rs.rand(n, 1)),
                                                 axis=1)
                pts[3*n:4*n, :] = np.concatenate((rs.rand(n, 1),
                                                  np.ones((n, 1)),
                                                  rs.rand(n, 1)),
                                                 axis=1)

                pts[4*n:5*n, :2] = rs.rand(n, 2)
                pts[5*n:6*n, :] = np.concatenate((rs.rand(n, 2),
                                                  np.ones((n, 1))),
                                                 axis=1)

        else:
            raise ValueError("Unknown sampling !")

        plotting = {
            'vertex_size': 80,
            'elevation': 15,
            'azimuth': 0,
            'distance': 7,
        }

        super(Cube, self).__init__(Xin=pts, k=10, gtype="Cube",
                                   plotting=plotting, **kwargs)
