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
    r"""Sphere sampled with the HEALPix scheme.

    Background information is found at :doc:`/background/spherical_samplings`.

    Parameters
    ----------
    subdivisions : int
        Number of recursive subdivisions. Known as ``order`` in HEALPix
        terminology, with ``nside=2**order`` and ``npix=12*4**order``.
    indexes : array_like of int
        Indexes of the pixels from which to build a graph. Useful to build a
        graph from a subset of the pixels, e.g., for partial sky observations.
    nest : bool
        Whether to assume NESTED or RING pixel ordering (default RING).
    kwargs : dict
        Additional keyword parameters are passed to :class:`NNGraph`.

    Attributes
    ----------
    signals : dict
        Vertex position as latitude ``'lat'`` in [-π/2,π/2] and longitude
        ``'lon'`` in [0,2π[.

    See Also
    --------
    SphereEquiangular, SphereGaussLegendre : based on quadrature theorems
    SphereIcosahedral : based on subdivided polyhedra
    SphereRandom : random uniform sampling

    Notes
    -----
    Edge weights are computed by :class:`NNGraph`. Gaussian kernel widths have
    been optimized for some combinations of resolutions `nlat` and number of
    neighbors `k` for the convolutions on the resulting graph to be maximally
    equivariant to rotation [2]_.

    References
    ----------
    .. [1] K. M. Gorski et al., HEALPix: a Framework for High Resolution
       Discretization and Fast Analysis of Data Distributed on the Sphere,
       2005.
    .. [2] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2019.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereHealpix(1, k=8)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax2 = fig.add_subplot(132, projection='3d')
    >>> ax3 = fig.add_subplot(133)
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = G.plot(ax=ax2)
    >>> G.set_coordinates('sphere', dim=2)
    >>> _ = G.plot(ax=ax3, indices=True)

    Vertex orderings:

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2)
    >>> graph = graphs.SphereHealpix(1, nest=False, k=8)
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot(indices=True, ax=axes[0], title='RING ordering')
    >>> graph = graphs.SphereHealpix(1, nest=True, k=8)
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot(indices=True, ax=axes[1], title='NESTED ordering')

    """

    def __init__(self, subdivisions=1, indexes=None, nest=False, **kwargs):
        hp = _import_hp()

        nside = hp.order2nside(subdivisions)
        self.subdivisions = subdivisions
        self.nest = nest

        if indexes is None:
            npix = hp.nside2npix(nside)
            indexes = np.arange(npix)

        x, y, z = hp.pix2vec(nside, indexes, nest=nest)
        coords = np.stack([x, y, z], axis=1)

        k = kwargs.pop('k', 20)
        try:
            kernel_width = kwargs.pop('kernel_width')
        except KeyError:
            try:
                kernel_width = _OPTIMAL_KERNEL_WIDTHS[k][nside]
            except KeyError:
                raise ValueError('No known optimal kernel width for {} '
                                 'neighbors and nside={}.'.format(k, nside))

        super(SphereHealpix, self).__init__(coords, k=k,
                                            kernel_width=kernel_width,
                                            **kwargs)

        lat, lon = hp.pix2ang(nside, indexes, nest=nest, lonlat=False)
        self.signals['lat'] = np.pi/2 - lat  # colatitude to latitude
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'subdivisions': self.subdivisions,
            'nest': self.nest,
        }
        attrs.update(super(SphereHealpix, self)._get_extra_repr())
        return attrs


# TODO: find an interpolation between nside and k (#neighbors).
_OPTIMAL_KERNEL_WIDTHS = {
    8: {
        1:    0.02500 * 32,  # extrapolated
        2:    0.02500 * 16,  # extrapolated
        4:    0.02500 * 8,  # extrapolated
        8:    0.02500 * 4,  # extrapolated
        16:   0.02500 * 2,  # extrapolated
        32:   0.02500,
        64:   0.01228,
        128:  0.00614,
        256:  0.00307,
        512:  0.00154,
        1024: 0.00077,
        2048: 0.00077 / 2,  # extrapolated
    },
    20: {
        1:    0.03185 * 32,  # extrapolated
        2:    0.03185 * 16,  # extrapolated
        4:    0.03185 * 8,  # extrapolated
        8:    0.03185 * 4,  # extrapolated
        16:   0.03185 * 2,  # extrapolated
        32:   0.03185,
        64:   0.01564,
        128:  0.00782,
        256:  0.00391,
        512:  0.00196,
        1024: 0.00098,
        2048: 0.00098 / 2,  # extrapolated
    },
    40: {
        1:    0.042432 * 32,  # extrapolated
        2:    0.042432 * 16,  # extrapolated
        4:    0.042432 * 8,  # extrapolated
        8:    0.042432 * 4,  # extrapolated
        16:   0.042432 * 2,  # extrapolated
        32:   0.042432,
        64:   0.021354,
        128:  0.010595,
        256:  0.005551,  # seems a bit off
        # 512:  0.003028,  # seems buggy
        512:  0.005551 / 2,  # extrapolated
        1024: 0.005551 / 4,  # extrapolated
        2048: 0.005551 / 8,  # extrapolated
    },
    60: {
        1:    0.051720 * 32,  # extrapolated
        2:    0.051720 * 16,  # extrapolated
        4:    0.051720 * 8,  # extrapolated
        8:    0.051720 * 4,  # extrapolated
        16:   0.051720 * 2,  # extrapolated
        32:   0.051720,
        64:   0.025403,
        128:  0.012695,
        256:  0.006351,
        # 512:  0.002493,  # seems buggy
        512:  0.006351 / 2,  # extrapolated
        1024: 0.006351 / 4,  # extrapolated
        2048: 0.006351 / 8,  # extrapolated
    },
}
