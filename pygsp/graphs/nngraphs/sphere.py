# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


class Sphere(NNGraph):
    r"""Randomly sampled hypersphere.

    Parameters
    ----------
    N : int
        Number of vertices (default = 300).
    dim : int
        Dimensionality of the space the hypersphere is embedded in (default = 3).
    radius : float
        Radius of the sphere (default = 2)
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sphere(100, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self, N=300, dim=3, radius=1, seed=None, **kwargs):

        self.dim = dim
        self.radius = radius
        self.seed = seed

        rs = np.random.RandomState(seed)
        coords = rs.normal(0, 1, (N, dim))
        coords *= radius / np.linalg.norm(coords, axis=1)[:, np.newaxis]

        plotting = {
            'vertex_size': 80,
        }

        super(Sphere, self).__init__(coords, plotting=plotting, **kwargs)

        if dim == 3:
            lat, lon = utils.xyz2latlon(*coords.T)
            self.signals['lat'] = lat
            self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'dim': self.dim,
            'radius': '{:.2e}'.format(self.diameter),
            'seed': self.seed
        }
        attrs.update(super(Sphere, self)._get_extra_repr())
        return attrs
