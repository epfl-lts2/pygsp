# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class Sphere(NNGraph):
    r"""Spherical-shaped graph (NN-graph).

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
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(10, 8))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> graphs.Sphere().plot(ax=ax)

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

        plotting = {
            'vertex_size': 80,
        }

        super(Sphere, self).__init__(Xin=pts, gtype='Sphere', k=10,
                                     plotting=plotting, **kwargs)
