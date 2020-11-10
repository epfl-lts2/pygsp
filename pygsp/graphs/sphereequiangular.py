# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp.graphs import Graph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereEquiangular(Graph):
    r"""Spherical-shaped graph using equirectangular sampling scheme.

    Parameters
    ----------
    bandwidth : int or list or tuple
        Resolution of the sampling scheme, corresponding to the bandwidth (latitude, latitude).
        Use a list or tuple to have a different resolution for latitude and longitude (default = 64)
    sampling : {'Driscoll-Heally', 'SOFT', 'Clenshaw-Curtis', 'Gauss-Legendre'}
        sampling scheme (default = 'SOFT')
        * Driscoll-Healy is the original sampling scheme of the sphere
        * SOFT is an upgraded version without the poles
        * Clenshaw-Curtis use the quadrature of its name to find the position of the latitude rings
        * Gauss-Legendre use the quadrature of its name to find the position of the latitude rings
        * Optimal Dimensionality guarranty the use of a minimum number of pixels, different for each latitude ring

    See Also
    --------
    SphereHealpix, SphereIcosahedron

    Notes
    ------
    Driscoll-Heally is the original sampling scheme of the sphere [1]
    SOFT is an updated sampling scheme, without the poles[2]
    Clenshaw-Curtis is [3]
    Gauss-Legendre is [4]
    The weight matrix is designed following [5]_

    References
    ----------
    [1] J. R. Driscoll et D. M. Healy, « Computing Fourier Transforms and Convolutions on the 2-Sphere »,
    Advances in Applied Mathematics, vol. 15, no. 2, pp. 202‑250, June 1994.
    [2] D. M. Healy, D. N. Rockmore, P. J. Kostelec, et S. Moore, « FFTs for the 2-Sphere-Improvements
    and Variations », Journal of Fourier Analysis and Applications, vol. 9, no. 4, pp. 341‑385, Jul. 2003.
    [3] D. Hotta and M. Ujiie, ‘A nestable, multigrid-friendly grid on a sphere for global spectral models
    based on Clenshaw-Curtis quadrature’, Q J R Meteorol Soc, vol. 144, no. 714, pp. 1382–1397, Jul. 2018.
    [4] J. Keiner et D. Potts, « Fast evaluation of quadrature formulae on the sphere »,
    Math. Comp., vol. 77, no. 261, pp. 397‑419, Jan. 2008.
    [5] P. Frossard and R. Khasanova, ‘Graph-Based Classification of Omnidirectional Images’,
    in 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), Venice, Italy, 2017, pp. 860–869.
    [6] Z. Khalid, R. A. Kennedy, et J. D. McEwen, « An Optimal-Dimensionality Sampling Scheme
    on the Sphere with Fast Spherical Harmonic Transforms », IEEE Transactions on Signal Processing,
    vol. 62, no. 17, pp. 4597‑4610, Sept. 2014.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G1 = graphs.SphereEquiangular(bandwidth=6, sampling='Driscoll-Healy')
    >>> G2 = graphs.SphereEquiangular(bandwidth=6, sampling='SOFT')
    >>> G3 = graphs.SphereEquiangular(bandwidth=6, sampling='Clenshaw-Curtis')
    >>> G4 = graphs.SphereEquiangular(bandwidth=6, sampling='Gauss-Legendre')
    >>> fig = plt.figure()
    >>> plt.subplots_adjust(wspace=1.)
    >>> ax1 = fig.add_subplot(221, projection='3d')
    >>> ax2 = fig.add_subplot(222, projection='3d')
    >>> ax3 = fig.add_subplot(223, projection='3d')
    >>> ax4 = fig.add_subplot(224, projection='3d')
    >>> _ = G1.plot(ax=ax1, title='Driscoll-Healy', vertex_size=10)
    >>> _ = G2.plot(ax=ax2, title='SOFT', vertex_size=10)
    >>> _ = G3.plot(ax=ax3, title='Clenshaw-Curtis', vertex_size=10)
    >>> _ = G4.plot(ax=ax4, title='Gauss-Legendre', vertex_size=10)
    >>> ax1.set_xlim([0, 1])
    >>> ax1.set_ylim([-1, 0.])
    >>> ax1.set_zlim([0.5, 1.])
    >>> ax2.set_xlim([0, 1])
    >>> ax2.set_ylim([-1, 0.])
    >>> ax2.set_zlim([0.5, 1.])
    >>> ax3.set_xlim([0, 1])
    >>> ax3.set_ylim([-1, 0.])
    >>> ax3.set_zlim([0.5, 1.])
    >>> ax4.set_xlim([0, 1])
    >>> ax4.set_ylim([-1, 0.])
    >>> ax4.set_zlim([0.5, 1.])

    """
    def __init__(self, size=(4, 8), poles=0, **kwargs):

        if isinstance(size, int):
            nlat, nlon = size, size
        else:
            nlat, nlon = size

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
