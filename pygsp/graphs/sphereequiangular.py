# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp.graphs import Graph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereEquiangular(Graph):
    r"""Sphere sampled with an equiangular scheme.

    The sphere is sampled as a grid whose vertices are separated by equal
    latitudinal and longitudinal angles.

    Background information is found at :doc:`/background/spherical_samplings`.

    Parameters
    ----------
    size : int or (int, int)
        Size of the discretization in latitude and longitude ``(nlat, nlon)``.
        ``nlat`` is the number of isolatitude (longitudinal) rings.
        ``nlon`` is the number of vertices (pixels) per ring.
        The total number of vertices is ``nlat*nlon``.
        ``nlat=nlon`` if only one number is given.
    poles : {0, 1, 2}
        Whether to sample 0, 1, or the 2 poles:
        0: nearest rings at ``dlat/2`` from both poles (``dlat=π/nlat``),
        1: ring at a pole, ring at ``dlat`` from other pole (``dlat=π/nlat``),
        2: rings at both poles (``dlat=π/(nlat-1)``).

    Attributes
    ----------
    signals : dict
        Vertex position as latitude ``'lat'`` in [-π/2,π/2] and longitude
        ``'lon'`` in [0,2π[.

    See Also
    --------
    SphereGaussLegendre : based on quadrature theorems
    SphereIcosahedral, SphereHealpix : based on subdivided polyhedra
    SphereRandom : random uniform sampling

    Notes
    ------
    Edge weights are computed as the reciprocal of the distances between
    vertices [6]_. This yields a graph convolution that is equivariant to
    longitudinal and latitudinal rotations, but not general rotations [7]_.

    References
    ----------
    .. [1] J. R. Driscoll et D. M. Healy, Computing Fourier Transforms and
       Convolutions on the 2-Sphere, 1994.
    .. [2] D. M. Healy et al., FFTs for the 2-Sphere-Improvements and
       Variations, 2003.
    .. [3] W. Skukowsky, A quadrature formula over the sphere with application
       to high resolution spherical harmonic analysis, 1986.
    .. [4] J. Keiner and D. Potts, Fast evaluation of quadrature formulae on
       the sphere, 2008.
    .. [5] J. D. McEwen and Y. Wiaux, A novel sampling theorem on the sphere,
       2011.
    .. [6] R. Khasanova and P. Frossard, Graph-Based Classification of
       Omnidirectional Images, 2017.
    .. [7] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2019.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereEquiangular()
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax2 = fig.add_subplot(132, projection='3d')
    >>> ax3 = fig.add_subplot(133)
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = G.plot(ax=ax2)
    >>> G.set_coordinates('sphere', dim=2)
    >>> _ = G.plot(ax=ax3, indices=True)

    Sampling of the poles:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> for i in range(3):
    ...     graph = graphs.SphereEquiangular(poles=i)
    ...     ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ...     _ = graph.plot(title=f'poles={i}', ax=ax)

    """
    def __init__(self, size=(4, 8), poles=0, **kwargs):

        if isinstance(size, int):
            nlat, nlon = size, size
        else:
            nlat, nlon = size

        if poles not in [0, 1, 2]:
            raise ValueError('poles must be 0, 1, or 2.')

        self.nlat = nlat
        self.nlon = nlon
        self.poles = poles
        npix = nlat*nlon

        lon = np.linspace(0, 2*np.pi, nlon, endpoint=False)
        lat = np.linspace(np.pi/2, -np.pi/2, nlat, endpoint=(poles == 2))
        if poles == 0:
            lat -= np.pi/2/nlat

        lat, lon = np.meshgrid(lat, lon, indexing='ij')
        lat, lon = lat.flatten(), lon.flatten()
        coords = np.stack(utils.latlon2xyz(lat, lon), axis=1)

        sources_vert = np.arange(0, npix-nlon)
        targets_vert = np.arange(nlon, npix)
        sources_horiz = np.arange(0, npix-1)
        targets_horiz = np.arange(1, npix)
        targets_horiz[nlon-1::nlon] -= nlon  # across meridian at 0
        sources = np.concatenate([sources_vert, sources_horiz])
        targets = np.concatenate([targets_vert, targets_horiz])

        distances = np.linalg.norm((coords[sources] - coords[targets]), axis=1)
        weights = 1 / distances

        adj = sparse.csr_matrix((weights, (sources, targets)), (npix, npix))
        adj = utils.symmetrize(adj, 'maximum')

        super(SphereEquiangular, self).__init__(adj, coords=coords, **kwargs)

        self.signals['lat'] = lat
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        return {
            'nlat': self.nlat,
            'nlon': self.nlon,
            'poles': self.poles,
        }
