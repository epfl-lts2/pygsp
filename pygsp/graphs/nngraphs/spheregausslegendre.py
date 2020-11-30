# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import roots_legendre

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereGaussLegendre(NNGraph):
    r"""Sphere sampled with a Gauss--Legendre scheme.

    The sphere is sampled at rings of constant latitude, where the latitudes
    are given by the zeros of the Legendre polynomial.

    Background information is found at :doc:`/background/spherical_samplings`.

    Parameters
    ----------
    nlat : int
        Number of rings of constant latitude.
    nlon : int or {'ecmwf-octahedral'}
        Number of vertices per ring for full grids, resulting in ``nlat*nlon``
        vertices. The default is ``2*nlat``.
        Reduced grids are named.
        If ``'ecmwf-octahedral'``, there are ``4*i+16`` vertices per ring,
        where ``i`` is the ring number from 1 (nearest to the poles) to
        ``nlat/2`` (nearest to the equator).
    kwargs : dict
        Additional keyword parameters are passed to :class:`NNGraph`.

    Attributes
    ----------
    signals : dict
        Vertex position as latitude ``'lat'`` in [-π/2,π/2] and longitude
        ``'lon'`` in [0,2π[.

    See Also
    --------
    SphereEquiangular : based on quadrature theorems
    SphereIcosahedral, SphereCubed, SphereHealpix :
        based on subdivided polyhedra
    SphereRandom : random uniform sampling

    Notes
    -----
    Edge weights are computed by :class:`NNGraph`. Gaussian kernel widths have
    however not been optimized for convolutions on the resulting graph to be
    maximally equivariant to rotation [8]_.

    The ECMWF's Integrated Forecast System (IFS) O320 grid is instantiated as
    ``SphereGaussLegendre(640, reduced='ecmwf-octahedral')`` [6]_ [7]_.

    References
    ----------
    .. [1] M. H. Payne, Truncation effects in geopotential modelling, 1971.
    .. [2] A. G. Doroshkevich et al., Gauss–Legendre sky pixelization (GLESP)
       for CMB maps, 2005.
    .. [3] N Schaeffer, Efficient spherical harmonic transforms aimed at
       pseudospectral numerical simulations, 2013.
    .. [4] J. Keiner and D. Potts, Fast evaluation of quadrature formulae on
       the sphere, 2008.
    .. [5] M. Hortal and A. J. Simmons, Use of reduced Gaussian grids in
       spectral models, 1991.
    .. [6] https://confluence.ecmwf.int/display/FCST/Introducing+the+octahedral+reduced+Gaussian+grid
    .. [7] https://confluence.ecmwf.int/display/OIFS/4.2+OpenIFS%3A+Octahedral+grid
    .. [8] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2019.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereGaussLegendre()
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax2 = fig.add_subplot(132, projection='3d')
    >>> ax3 = fig.add_subplot(133)
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = G.plot(ax=ax2)
    >>> G.set_coordinates('sphere', dim=2)
    >>> _ = G.plot(ax=ax3, indices=True)

    Full and reduced grids:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> graph = graphs.SphereGaussLegendre(20, 2*20+16)
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot('C0', 20, edges=False, ax=ax)
    >>> graph = graphs.SphereGaussLegendre(20, 'ecmwf-octahedral')
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot('C1', 20, edges=False, ax=ax)
    >>> _ = ax.set_title('Full and reduced grids')

    """

    def __init__(self, nlat=4, nlon=None, **kwargs):

        if nlon is None:
            nlon = 2 * nlat

        self.nlat = nlat
        self.nlon = nlon

        z = -roots_legendre(nlat)[0]
        lat_ = np.arcsin(z)

        if type(nlon) is not str:
            lon_ = np.linspace(0, 2*np.pi, nlon, endpoint=False)
            lat, lon = np.meshgrid(lat_, lon_, indexing='ij')
            lat, lon = lat.flatten(), lon.flatten()

        elif nlon == 'ecmwf-octahedral':
            odd = nlat % 2
            npix = nlat*(nlat+18) + odd
            lon = np.empty(npix)
            lat = np.empty(npix)
            i = 0
            for ring in range(nlat//2 + odd):
                npix_per_ring = 4*(ring+1) + 16
                lon_ = np.linspace(0, 2*np.pi, npix_per_ring, endpoint=False)
                lat[i:i+npix_per_ring] = lat_[ring]
                lon[i:i+npix_per_ring] = lon_
                lat[npix-i-npix_per_ring:npix-i] = -lat_[ring]
                lon[npix-i-npix_per_ring:npix-i] = lon_
                i += npix_per_ring

        elif nlon == 'glesp':  # Newer GLESP-pol (grS) [arXiv:0904.2517].
            # npix_per_ring = 4*(ring+1) + 10
            raise NotImplementedError
        elif nlon == 'glesp-equal-area':  # [arXiv:astro-ph/0305537].
            # All have about the same area as the square pixels at the equator.
            dlat = lat_[nlat//2] - lat_[nlat//2-1]
            dlon = 2*np.pi / round(2*np.pi/dlat)
            area = dlat * dlon
            npix = np.round(2*np.pi * np.sqrt(1-z**2) / area)
            raise NotImplementedError('Must be checked and fixed.')

        else:
            raise ValueError('Unexpected nlon={}.'.format(nlon))

        coords = np.stack(utils.latlon2xyz(lat, lon), axis=1)

        super(SphereGaussLegendre, self).__init__(coords, **kwargs)

        self.signals['lat'] = lat
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'nlat': self.nlat,
            'nlon': self.nlon,
        }
        attrs.update(super(SphereGaussLegendre, self)._get_extra_repr())
        return attrs
