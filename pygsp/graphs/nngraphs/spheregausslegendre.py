# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereGaussLegendre(NNGraph):
    r"""Sphere sampled with a Gauss-Legendre scheme.
    """

    def __init__(self, nrings=10, reduced=False, **kwargs):

        self.nrings = nrings
        self.reduced = reduced

        # TODO: docstring states that degree > 100 may be problematic.
        z = -np.polynomial.legendre.leggauss(nrings)[0]
        lat_ = np.arcsin(z)

        if reduced is False:
            lon_ = np.linspace(0, 2*np.pi, 2*nrings, endpoint=False)
            lat, lon = np.meshgrid(lat_, lon_, indexing='ij')
            lat, lon = lat.flatten(), lon.flatten()

        elif reduced == 'ecmwf-octahedral':
            odd = nrings % 2
            npix = nrings*(nrings+18) + odd
            lon = np.empty(npix)
            lat = np.empty(npix)
            i = 0
            for ring in range(nrings//2 + odd):
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
            dlat = lat_[nrings//2] - lat_[nrings//2-1]
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
            'nrings': self.nrings,
            'reduced': self.reduced,
        }
        attrs.update(super(SphereGaussLegendre, self)._get_extra_repr())
        return attrs
