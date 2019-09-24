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
    >>> G = graphs.SphereHealpix(nside=4)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self, indexes=None, nside=32, nest=True, kernel_width=None, n_neighbors=50, **kwargs):
        hp = _import_hp()
        self.nside = nside
        self.nest = nest
        npix = hp.nside2npix(nside)
        if indexes is None:
            indexes = np.arange(npix)
        x, y, z = hp.pix2vec(nside, indexes, nest=nest)
        self.lat, self.lon = hp.pix2ang(nside, indexes, nest=nest, lonlat=False)
        coords = np.vstack([x, y, z]).transpose()
        coords = np.asarray(coords, dtype=np.float32)
        if n_neighbors is None:
            n_neighbors = 6 if Nside==1 else 8
            
        self.opt_std = dict()
        self.opt_std[20] =  {
                    32:   0.03185,
                    64:   0.01564,
                    128:  0.00782,
                    256:  0.00391,
                    512:  0.00196,
                    1024: 0.00098,
        }
        self.opt_std[40] =  {
                    32:   0.042432,
                    64:   0.021354,
                    128:  0.010595,
                    256:  0.005551,  # seems a bit off
                    #512:  0.003028,  # seems buggy
                    512:  0.005551 / 2,  # extrapolated
                    1024: 0.005551 / 4,  # extrapolated
        }
        self.opt_std[60] =  {
                    32:   0.051720,
                    64:   0.025403,
                    128:  0.012695,
                    256:  0.006351,
                    #512:  0.002493,  # seems buggy
                    512:  0.006351 / 2,  # extrapolated
                    1024: 0.006351 / 4,  # extrapolated
        }
        self.opt_std[8] = {
                    32:   0.02500,
                    64:   0.01228,
                    128:  0.00614,
                    256:  0.00307,
                    512:  0.00154,
                    1024: 0.00077,
        }
        try:
            kernel_dict = self.opt_std[n_neighbors]
        except:
            raise ValueError('No sigma for number of neighbors {}'.format(n_neighbors))
        try:
            kernel_width = kernel_dict[nside]
        except:
            raise ValueError('Unknown sigma for nside {}'.format(nside))
        ## TODO: check std
    
        ## TODO: n_neighbors in function of Nside
        if len(indexes) <= n_neighbors:
            n_neighbors = len(indexes)-1
        
        plotting = {
            'vertex_size': 80,
            "limits": np.array([-1, 1, -1, 1, -1, 1])
        }
        super(SphereHealpix, self).__init__(features=coords, k=n_neighbors,
                                     kernel_width=kernel_width, plotting=plotting, **kwargs)
        
