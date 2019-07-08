# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp.graphs import Graph # prevent circular import in Python < 3.5


def _import_hp():
    try:
        import healpy as hp
    except Exception as e:
        raise ImportError('Cannot import healpy. Choose another graph '
                          'or try to install it with '
                          'conda install healpy. '
                          'Original exception: {}'.format(e))
    return hp

class SphereEquiangular(Graph):
    r"""Spherical-shaped graph using equirectangular sampling scheme.

    Parameters
    ----------
    bw : int or list or tuple
        Resolution of the sampling scheme, corresponding to the bandwidth.
        Use a list or tuple to have a different resolution for latitude and longitude (default = 64)
    sptype : string
        sampling type (default = 'SOFT')
        * DH original Driscoll-Healy
        * SOFT equiangular without poles
        * CC use of Clenshaw-Curtis quadrature
        * GL use of Gauss-Legendre quadrature
        * OD optimal dimensionality
    dist : string
        type of distance use to compute edge weights, euclidean or geodesic (default = 'euclidean')
    cylinder : bool
        adapt the grid on a cylinder

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereEquiangular(bw=8, sptype='SOFT')
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """
    def __init__(self, bw=64, sptype='DH', dist='euclidean', cylinder=False, **kwargs):
        if isinstance(bw, int):
            bw = (bw, bw)
        elif len(bw)>2:
            raise ValueError('Cannot have more than two bandwidths')
        self.bw = bw
        self.sptype = sptype
        if sptype not in ['DH', 'SOFT', 'CC', 'GL', 'OD']:
            raise ValueError('Unknown sampling type:' + sptype)
        if dist not in ['geodesic', 'euclidean']:
            raise ValueError('Unknown distance type value:' + dist)

        ## sampling and coordinates calculation
        if sptype is 'DH':
            beta = np.arange(2 * bw[0]) * np.pi / (2. * bw[0])  # Driscoll-Heally
            alpha = np.arange(2 * bw[1]) * np.pi / bw[1]
        elif sptype is 'SOFT':  # SO(3) Fourier Transform optimal
            beta = np.pi * (2 * np.arange(2 * bw[0]) + 1) / (4. * bw[0])
            alpha = np.arange(2 * bw[1]) * np.pi / bw[1]
        elif sptype == 'CC':  # Clenshaw-Curtis
            beta = np.linspace(0, np.pi, 2 * bw[0] + 1)
            alpha = np.linspace(0, 2 * np.pi, 2 * bw[1] + 2, endpoint=False)
        elif sptype == 'GL':  # Gauss-legendre
            try:
                from numpy.polynomial.legendre import leggauss
            except:
                raise ImportError("cannot import legendre quadrature from numpy."
                                  "Choose another sampling type or upgrade numpy.")
            x, _ = leggauss(bw[0] + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
            beta = np.arccos(x)
            alpha = np.arange(2 * bw[1] + 2) * np.pi / (bw[1] + 1)
        if sptype == 'OD':  # Optimal Dimensionality
            theta, phi = np.zeros(4*bw[0]**2), np.zeros(4*bw[0]**2)
            index=0
            #beta = np.pi * (2 * np.arange(2 * bw) + 1) / (4. * bw)
            beta = np.pi * ((np.arange(2 * bw[0] + 1)%2)*(4*bw[0]-1)+np.arange(2 * bw[0] + 1)*-
                    1**(np.arange(2 * bw[0] + 1)%2)) / (4 * bw[0] - 1)
            for i in range(2*bw[0]):
                alpha = 2 * np.pi * np.arange(2 * i + 1) / (2 * i + 1)
                end = len(alpha)
                theta[index:index+end], phi[index:index+end] = np.repeat(beta[i], end), alpha
                index += end
        else:
            theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
        self.lat, self.lon = theta.shape
        if cylinder:
            ct = theta.flatten() * 2 * bw[1] / np.pi
            st = 1
        else:
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

        ## neighbors and weight matrix calculation
        def south(x):
            if x >= self.npix - self.lat:
                return (x + self.lat//2)%self.lat + self.npix - self.lat
            return x + self.lon

        def north(x):
            if x < self.lat:
                return (x + self.lat//2)%self.lat
            return x - self.lon

        def west(x):
            if x%(self.lon)<1:
                try:
                    assert x//self.lat == (x-1+self.lon)//self.lat
                except:
                    raise
                x += self.lon
            else:
                try:
                    assert x//self.lat == (x-1)//self.lat
                except:
                    raise
            return x - 1

        def east(x):
            if x%(self.lon)>=self.lon-1:
                try:
                    assert x//self.lat == (x+1-self.lon)//self.lat
                except:
                    raise
                x -= self.lon
            else:
                try:
                    assert x//self.lat == (x+1)//self.lat
                except:
                    raise
            return x + 1

        col_index=[]
        for ind in range(self.npix):
            # if neighbors==8:
            #     neighbor = [south(west(ind)), west(ind), north(west(ind)), north(ind),
            #                 north(east(ind)), east(ind), south(east(ind)), south(ind)]
            # elif neighbors==4:
            neighbor = [west(ind), north(ind), east(ind), south(ind)]
            # else:
            #     neighbor = []
            col_index += neighbor
        col_index = np.asarray(col_index)
        row_index = np.repeat(np.arange(self.npix), 4)

        keep = (col_index < self.npix)
        keep &= (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]

        if dist=='geodesic':
            hp = _import_hp()
            distances = np.zeros(len(row_index))
            for i, (pos1, pos2) in enumerate(zip(coords[row_index], coords[col_index])):
                d1, d2 = hp.rotator.vec2dir(pos1.T, lonlat=False).T, hp.rotator.vec2dir(pos2.T, lonlat=False).T
                distances[i] = hp.rotator.angdist(d1, d2, lonlat=False)
        else:
            distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)

        # Compute similarities / edge weights.
        kernel_width = np.mean(distances)

        # weights = np.exp(-distances / (2 * kernel_width))
        weights = 1/distances

        W = sparse.csr_matrix(
            (weights, (row_index, col_index)), shape=(self.npix, self.npix), dtype=np.float32)

        plotting = {"limits": np.array([-1, 1, -1, 1, -1, 1])}
        super(SphereEquiangular, self).__init__(adjacency=W, coords=coords,
                                     plotting=plotting, **kwargs)

