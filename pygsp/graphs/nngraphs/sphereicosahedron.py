# -*- coding: utf-8 -*-

import numpy as np
import scipy

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5
from pygsp import utils


def _import_trimesh():
    try:
        import trimesh
    except Exception as e:
        raise ImportError('Cannot import trimesh. Choose another graph '
                          'or try to install it with '
                          'conda or pip install trimesh. '
                          'Original exception: {}'.format(e))
    return trimesh


class SphereIcosahedron(NNGraph):
    r"""Spherical-shaped graph based on the projection of the icosahedron (NN-graph).
    Code inspired by Max Jiang [https://github.com/maxjiang93/ugscnn/blob/master/meshcnn/mesh.py]

    Parameters
    ----------
    level : int
        Resolution of the sampling scheme, or how many times the faces are divided (default = 5)
    sampling : string
        What the pixels represent. Either a vertex or a face (default = 'vertex')

    See Also
    --------
    SphereDodecahedron, SphereHealpix, SphereEquiangular

    Notes
    ------
    The icosahedron is the dual of the dodecahedron. Thus the pixels in this graph represent either the vertices \
    of the icosahedron, or the faces of the dodecahedron.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> G = graphs.SphereIcosahedron(level=1)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """
    def __init__(self, subdivisions=2, dual=False, **kwargs):

        self.subdivisions = subdivisions
        self.dual = dual

        # Vertices as the corners of three orthogonal golden planes.
        φ = scipy.constants.golden_ratio
        vertices = np.array([
            [-1, φ, 0], [1, φ, 0], [-1, -φ, 0], [1, -φ, 0],
            [0, -1, φ], [0, 1, φ], [0, -1, -φ], [0, 1, -φ],
            [φ, 0, -1], [φ, 0, 1], [-φ, 0, -1], [-φ, 0, 1],
        ]) / np.sqrt(φ**2+1)
        faces = np.array([
            # Faces around vertex 0.
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            # Adjacent faces.
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            # Faces around vertex 3.
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            # Adjacent faces.
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ])

        trimesh = _import_trimesh()
        mesh = trimesh.Trimesh(vertices, faces)

        def normalize(vertices):
            """Project the vertices on the sphere."""
            vertices /= np.linalg.norm(vertices, axis=1)[:, None]

        for _ in range(subdivisions):
            mesh = mesh.subdivide()
            # TODO: shall we project between subdivisions? Some do, some don't.
            # Projecting pushes points away from the 12 base vertices, which
            # may make the point density more uniform.
            # See "A Comparison of Popular Point Configurations on S^2".
            normalize(mesh.vertices)

        if not dual:
            vertices = mesh.vertices
        else:
            vertices = mesh.vertices[mesh.faces].mean(axis=1)
            normalize(vertices)

        super(SphereIcosahedron, self).__init__(vertices, **kwargs)

        lat, lon = utils.xyz2latlon(*vertices.T)
        self.signals['lat'] = lat
        self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'subdivisions': self.subdivisions,
            'dual': self.dual,
        }
        attrs.update(super(SphereIcosahedron, self)._get_extra_repr())
        return attrs
