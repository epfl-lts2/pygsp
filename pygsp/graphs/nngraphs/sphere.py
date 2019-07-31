# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class Sphere(NNGraph):
    r"""Spherical-shaped graph (NN-graph).

    Parameters
    ----------
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    diameter : float
        Radius of the sphere (default = 2)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sphere(nb_pts=100, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self,
                 nb_pts=300,
                 nb_dim=3,
                 diameter=2,
                 sampling='random',
                 seed=None,
                 **kwargs):

        self.diameter = diameter
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling
        self.seed = seed

        if self.sampling == 'random':

            rs = np.random.RandomState(seed)
            pts = rs.normal(0, 1, (self.nb_pts, self.nb_dim))

            for i in range(self.nb_pts):
                pts[i] /= np.linalg.norm(pts[i])
                pts[i] *= (diameter / 2)

        else:

            raise ValueError('Unknown sampling {}'.format(sampling))

        plotting = {
            'vertex_size': 80,
        }

        super(Sphere, self).__init__(pts, k=10, plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        attrs = {'diameter': '{:.2e}'.format(self.diameter),
                 'nb_pts': self.nb_pts,
                 'nb_dim': self.nb_dim,
                 'sampling': self.sampling,
                 'seed': self.seed}
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
