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


class SphereOptimalDimensionality(NNGraph):
    r"""Spherical-shaped graph using optimal dimensionality sampling scheme (NN-graph).

    Parameters
    ----------
    bandwidth : int
        Resolution of the sampling scheme, corresponding to the number of latitude rings (default = 64)
    distance_type : {'euclidean', 'geodesic'}
        type of distance use to compute edge weights (default = 'euclidean')

    See Also
    --------
    SphereEquiangular, SphereHealpix, SphereIcosahedron

    Notes
    ------
    The optimal dimensionality[1]_ sampling scheme consists on `\mathtt{bandwidth}` latitude rings equispaced.
    The number of longitude pixels is different for each rings, and correspond to the number of spherical harmonics \
    for each mode.
    The number of pixels is then only `2*\mathtt{bandwidth}`

    References
    ----------
    [1] Z. Khalid, R. A. Kennedy, et J. D. McEwen, « An Optimal-Dimensionality Sampling Scheme
    on the Sphere with Fast Spherical Harmonic Transforms », IEEE Transactions on Signal Processing,
    vol. 62, no. 17, pp. 4597‑4610, Sept. 2014.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereOptimalDimensionality(bandwidth=8)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """
    def __init__(self, bandwidth=64, distance_type='euclidean', **kwargs):
        self.bandwidth = bandwidth
        if distance_type not in ['geodesic', 'euclidean']:
            raise ValueError('Unknown distance type value:' + distance_type)

        ## sampling and coordinates calculation
        theta, phi = np.zeros(4*bandwidth**2), np.zeros(4*bandwidth**2)
        index=0
        beta = np.pi * ((np.arange(2 * bandwidth + 1)%2)*(4*bandwidth-1)+np.arange(2 * bandwidth + 1)*-
                1**(np.arange(2 * bandwidth + 1)%2)) / (4 * bandwidth - 1)
        for i in range(2*bandwidth):
            alpha = 2 * np.pi * np.arange(2 * i + 1) / (2 * i + 1)
            end = len(alpha)
            theta[index:index+end], phi[index:index+end] = np.repeat(beta[i], end), alpha
            index += end
        self.lat, self.lon = theta-np.pi/2, phi
        self.bwlat, self.bwlon = theta.shape
        ct = np.cos(theta).flatten()
        st = np.sin(theta).flatten()
        cp = np.cos(phi).flatten()
        sp = np.sin(phi).flatten()
        x = st * cp
        y = st * sp
        z = ct
        coords = np.vstack([x, y, z]).T
        coords = np.asarray(coords, dtype=np.float32)
        self.npix = len(coords)

        plotting = {"limits": np.array([-1, 1, -1, 1, -1, 1])}
        super(SphereOptimalDimensionality, self).__init__(coords, k=4, plotting=plotting, **kwargs)
