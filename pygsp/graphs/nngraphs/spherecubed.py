# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


class SphereCubed(NNGraph):
    r"""Sphere sampled as a subdivided cube.

    Background information is found at :doc:`/background/spherical_samplings`.

    Parameters
    ----------
    subdivisions : int
        Number of edges the cube's edges are divided into, resulting in
        ``6*subdivisions**2`` vertices.
    spacing : {'equiangular', 'equidistant'}
        Whether the vertices (on a cube's face) are spaced by equal angles (as
        in [4]_) or equal distances (as in [1]_).
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
    SphereHealpix, SphereIcosahedral : based on subdivided polyhedra
    SphereRandom : random uniform sampling

    Notes
    -----
    Edge weights are computed by :class:`NNGraph`. Gaussian kernel widths have
    however not been optimized for convolutions on the resulting graph to be
    maximally equivariant to rotation [7]_.

    References
    ----------
    .. [1] R. Sadourny, Conservative finite-difference approximations of the
       primitive equations on quasi-uniform spherical grids, 1972.
    .. [2] EM O'Neill, RE Laubscher, Extended studies of a quadrilateralized
       spherical cube Earth data base, 1976.
    .. [3] R. A. White, S. W. Stemwedel, The quadrilateralized spherical cube
       and quad-tree for all sky data, 1992.
    .. [4] C. Ronchi, R. Iacono, P. Paolucci, The “Cubed-Sphere:” a new method
       for the solution of partial differential equations in spherical
       geometry, 1995.
    .. [5] W. M. Putmana, S.-J. Lin, Finite-volume transport on various
       cubed-sphere grids, 2007.
    .. [6] https://en.wikipedia.org/wiki/Quadrilateralized_spherical_cube
    .. [7] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2019.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SphereCubed()
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(131)
    >>> ax2 = fig.add_subplot(132, projection='3d')
    >>> ax3 = fig.add_subplot(133)
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = G.plot(ax=ax2)
    >>> G.set_coordinates('sphere', dim=2)
    >>> _ = G.plot(ax=ax3, indices=True)

    Equiangular and equidistant spacings:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> graph = graphs.SphereCubed(4, 'equiangular')
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot('C0', edges=False, ax=ax)
    >>> graph = graphs.SphereCubed(4, 'equidistant')
    >>> graph.set_coordinates('sphere', dim=2)
    >>> _ = graph.plot('C1', edges=False, ax=ax)
    >>> _ = ax.set_title('Equiangular and equidistant spacings')

    """

    def __init__(self, subdivisions=3, spacing='equiangular', **kwargs):

        self.subdivisions = subdivisions
        self.spacing = spacing

        def linspace(interval):
            x = np.linspace(-interval, interval, subdivisions, endpoint=False)
            x += interval / subdivisions
            return x

        a = np.sqrt(3) / 3
        if spacing == 'equidistant':
            x = linspace(a)
            y = linspace(a)
        elif spacing == 'equiangular':
            x = a * np.tan(linspace(np.pi/4))
            y = a * np.tan(linspace(np.pi/4))

        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        z = a * np.ones_like(x)

        coords = np.concatenate([
            [-y, x, z],   # North face.
            [z, x, y],    # Equatorial face centered at longitude 0.
            [-x, z, y],   # Equatorial face centered at longitude 90°.
            [-z, -x, y],  # Equatorial face centered at longitude 180°.
            [x, -z, y],   # Equatorial face centered at longitude 270°.
            [y, x, -z],   # South face.
        ], axis=1).T

        coords /= np.linalg.norm(coords, axis=1)[:, np.newaxis]

        super().__init__(coords, **kwargs)

        lat, lon = utils.xyz2latlon(*coords.T)
        self.signals['lat'] = lat
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'subdivisions': self.subdivisions,
            'spacing': self.spacing,
        }
        attrs.update(super(SphereCubed, self)._get_extra_repr())
        return attrs
