# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5

def _import_hp():
    try:
        import healpy as hp
    except Exception as e:
        raise ImportError('Cannot import healpy. Choose another graph '
                          'or try to install it with '
                          'conda install healpy. '
                          'Original exception: {}'.format(e))
    return hp


class SphereHealpix(NNGraph):
    r"""Spherical-shaped graph using HEALPix sampling scheme (NN-graph).

    Parameters
    ----------
    Nside : int
        Resolution of the sampling scheme. It should be a power of 2 (default = 1024)
    nest : bool
        ordering of the pixels (default = True)

    See Also
    --------
    SphereEquiangular, SphereIcosahedron

    Notes
    -----
    This graph us based on the HEALPix[1]_ sampling scheme mainly used by the cosmologist.
    Heat Kernel Distance is used to find its weight matrix.

    References
    ----------
    [1] K. M. Gorski et al., « HEALPix -- a Framework for High Resolution Discretization,
    and Fast Analysis of Data Distributed on the Sphere », ApJ, vol. 622, nᵒ 2, p. 759‑771, avr. 2005.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereHealpix(Nside=4)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self, indexes=None, Nside=32, nest=True, kernel_width=None, n_neighbors=None, **kwargs):
        hp = _import_hp()
        self.Nside = Nside
        self.nest = nest
        npix = hp.nside2npix(Nside)
        if indexes is None:
            indexes = np.arange(npix)
        x, y, z = hp.pix2vec(Nside, indexes, nest=nest)
        self.lat, self.lon = hp.pix2ang(Nside, indexes, nest=nest, lonlat=False)
        coords = np.vstack([x, y, z]).transpose()
        coords = np.asarray(coords, dtype=np.float32)
        ## TODO: n_neighbors in function of Nside
        if n_neighbors is None:
            n_neighbors = 6 if Nside==1 else 8
            if Nside>=4:
                n_neighbors = 50
            elif Nside == 2:
                n_neighbors = 47
            else:
                n_neighbors = 11
        if len(indexes)<50:
            n_neighbors = len(indexes)-1
        ## TODO: find optimal sigmas (for n_neighbors = 50)
        """opt_std =  {1:1.097324009878543,
                    2:1.097324042581347,
                    4: 0.5710655156439823,
                    8: 0.28754191240507265,
                    16: 0.14552024595543614,
                    32: 0.07439700765663292,
                    64: 0.03654101726025044,
                    128: 0.018262391329213392,
                    256: 0.009136370875837834,
                    512: 0.004570016186845779,
                    1024: 0.0022857004460788742,}
        """
        opt_std = {1:1.097324009878543,
                   2:1.097324042581347,
                   4: 0.5710655156439823,
                   8: 0.28754191240507265,
                   16: 0.14552024595543614,
                   32: 0.05172026,      ### from nside=32 on it was obtained by equivariance error minimization
                   64: 0.0254030519,
                   128: 0.01269588289,
                   256: 0.00635153921,
                   512: 0.002493215645,}

            try:
                kernel_width = opt_std[Nside]
            except:
                raise ValueError('Unknown sigma for nside>32')
        ## TODO: check std
        plotting = {
            'vertex_size': 80,
            "limits": np.array([-1, 1, -1, 1, -1, 1])
        }
        super(SphereHealpix, self).__init__(features=coords, k=n_neighbors,
                                     kernel_width=kernel_width, plotting=plotting, **kwargs)
        
