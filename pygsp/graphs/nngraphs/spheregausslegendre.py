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
        Number of isolatitude (longitudinal) rings.
    reduced : {False, 'ecmwf-octahedral'}
        If ``False``, there are ``2*nlat`` pixels per ring.
        If ``'ecmwf-octahedral'``, there are ``4*i+16`` pixels per ring, where
        ``i`` is the ring number from 1 (nearest to the poles) to ``nlat/2``
        (nearest to the equator).
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
    SphereIcosahedral, SphereHealpix : based on subdivided polyhedra
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

    """

    def __init__(self, nlat=4, reduced=False, **kwargs):

        self.nlat = nlat
        self.reduced = reduced

        z = -roots_legendre(nlat)[0]
        lat_ = np.arcsin(z)

        if reduced is False:
            lon_ = np.linspace(0, 2*np.pi, 2*nlat, endpoint=False)
            lat, lon = np.meshgrid(lat_, lon_, indexing='ij')
            lat, lon = lat.flatten(), lon.flatten()

        elif reduced == 'ecmwf-octahedral':
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

        elif reduced == 'glesp':  # Newer GLESP-pol (grS) [arXiv:0904.2517].
            # npix_per_ring = 4*(ring+1) + 10
            raise NotImplementedError
        elif reduced == 'glesp-equal-area':  # [arXiv:astro-ph/0305537].
            # All have about the same area as the square pixels at the equator.
            dlat = lat_[nlat//2] - lat_[nlat//2-1]
            dlon = 2*np.pi / round(2*np.pi/dlat)
            area = dlat * dlon
            npix = np.round(2*np.pi * np.sqrt(1-z**2) / area)
            raise NotImplementedError('Must be checked and fixed.')

        else:
            raise ValueError('Unexpected reduced={}.'.format(reduced))

        coords = np.stack(utils.latlon2xyz(lat, lon), axis=1)

        super(SphereGaussLegendre, self).__init__(coords, **kwargs)

        self.signals['lat'] = lat
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'nlat': self.nlat,
            'reduced': self.reduced,
        }
        attrs.update(super(SphereGaussLegendre, self)._get_extra_repr())
        return attrs
