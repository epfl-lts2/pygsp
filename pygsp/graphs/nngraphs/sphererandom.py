# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereRandom(NNGraph):
    r"""Random uniform sampling of an hypersphere.

    Parameters
    ----------
    N : int
        Number of vertices (default = 300).
    dim : int
        Dimension of the space the hypersphere is embedded in.
    seed : int
        Seed for the random number generator (for reproducible graphs).
    kwargs : dict
        Additional keyword parameters are passed to :class:`NNGraph`.

    Attributes
    ----------
    signals : dict
        Vertex position as latitude ``'lat'`` in [-π/2,π/2] and longitude
        ``'lon'`` in [0,2π[.

    See Also
    --------
    SphereEquiangular, SphereGaussLegendre : based on quadrature theorems
    SphereIcosahedral, SphereHealpix : based on subdivided polyhedra
    CubeRandom : randomly sampled cube

    References
    ----------
    .. [1] http://mathworld.wolfram.com/HyperspherePointPicking.html
    .. [2] J. S. Hicks and R. F. Wheeling, An Efficient Method for Generating
       Uniformly Distributed Points on the Surface of an n-Dimensional Sphere,
       1959.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereRandom(100, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax2 = fig.add_subplot(132, projection='3d')
    >>> ax3 = fig.add_subplot(133)
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = G.plot(ax=ax2)
    >>> G.set_coordinates('sphere', dim=2)
    >>> _ = G.plot(ax=ax3, indices=True)

    """

    def __init__(self, N=300, dim=3, seed=None, **kwargs):

        self.dim = dim
        self.seed = seed

        rs = np.random.RandomState(seed)
        coords = rs.normal(0, 1, (N, dim))
        coords /= np.linalg.norm(coords, axis=1)[:, np.newaxis]

        plotting = {
            'vertex_size': 80,
        }

        super(SphereRandom, self).__init__(coords, plotting=plotting, **kwargs)

        if dim == 3:
            lat, lon = utils.xyz2latlon(*coords.T)
            self.signals['lat'] = lat
            self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'dim': self.dim,
            'seed': self.seed,
        }
        attrs.update(super(SphereRandom, self)._get_extra_repr())
        return attrs
