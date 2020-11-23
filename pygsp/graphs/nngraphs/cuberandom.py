# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class CubeRandom(NNGraph):
    r"""Random uniform sampling of a cube.

    Parameters
    ----------
    N : int
        Number of vertices (default = 300). Will be rounded to a multiple of 6,
        for each face to have the same number of vertices.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    See Also
    --------
    Sensor : randomly sampled square
    SphereRandom : randomly sampled hypersphere

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.CubeRandom(seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=0.5)
    >>> _ = G.plot(ax=ax2)

    """

    def __init__(self, N=300, seed=None, **kwargs):

        self.seed = seed

        n = N // 6
        rs = np.random.RandomState(seed)
        coords = rs.uniform(0, 1, (6*n, 3))

        coords[0*n:1*n, 0] = np.zeros(n)  # face 1
        coords[1*n:2*n, 0] = np.ones(n)  # face 2
        coords[2*n:3*n, 1] = np.zeros(n)  # face 3
        coords[3*n:4*n, 1] = np.ones(n)  # face 4
        coords[4*n:5*n, 2] = np.zeros(n)  # face 5
        coords[5*n:6*n, 2] = np.ones(n)  # face 6

        plotting = {
            'vertex_size': 80,
            'elevation': 15,
            'azimuth': 0,
            'distance': 9,
        }

        super(CubeRandom, self).__init__(coords, plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        attrs = {'seed': self.seed}
        attrs.update(super(CubeRandom, self)._get_extra_repr())
        return attrs
