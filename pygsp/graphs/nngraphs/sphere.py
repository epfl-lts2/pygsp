# -*- coding: utf-8 -*-

from . import NNGraph

import numpy as np


class Sphere(NNGraph):
    r"""
    Creates a spherical-shaped graph.

    Parameters
    ----------
    radius : flaot
        Radius of the sphere (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : sting
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')

    Examples
    --------
    >>> from pygsp import graphs
    >>> radius = 5
    >>> G = graphs.Sphere(radius=radius)

    """

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling='random', **kwargs):
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.sampling == 'random':
            pts = np.random.normal(0, 1, (self.nb_pts, self.nb_dim))

            for i in range(self.nb_pts):
                pts[i] /= np.linalg.norm(pts[i])
        else:
            raise ValueError('Unknow sampling!')

        super(Sphere, self).__init__(Xin=pts, gtype='Sphere', k=10, **kwargs)
