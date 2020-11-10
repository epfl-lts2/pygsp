# -*- coding: utf-8 -*-

import numpy as np

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
    def __init__(self, level=5, sampling='vertex', **kwargs):

        if sampling not in ['vertex', 'face']:
            raise ValueError('Unknown sampling value:' + sampling)
        PHI = (1 + np.sqrt(5))/2
        radius = np.sqrt(PHI**2+1)
        coords = [-1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI, 0,
                  0, -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI,
                  PHI, 0, -1, PHI, 0, 1, -PHI, 0, -1, -PHI, 0, 1]
        coords = np.reshape(coords, (-1,3))
        coords = coords/radius
        faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
        self.faces = np.reshape(faces, (20, 3))
        self.level = level
        self.intp = None

        coords = self._upward(coords, self.faces)
        self.coords = coords

        trimesh = _import_trimesh()
        mesh = trimesh.Trimesh(coords, faces)

        def normalize(vertices):
            """Project the vertices on the sphere."""
            vertices /= np.linalg.norm(vertices, axis=1)[:, None]

        for i in range(level):
            mesh = mesh.subdivide()
            normalize(mesh.vertices)

        if sampling=='face':
            self.coords = self.coords[self.faces].mean(axis=1)

        self.lat, self.lon = utils.xyz2latlon()

        self.npix = len(self.coords)
        self.nf = 20 * 4**self.level
        self.ne = 30 * 4**self.level
        self.nv = self.ne - self.nf + 2
        self.nv_prev = int((self.ne / 4) - (self.nf / 4) + 2)
        self.nv_next = int((self.ne * 4) - (self.nf * 4) + 2)

        plotting = {
            'vertex_size': 80,
            "limits": np.array([-1, 1, -1, 1, -1, 1])
        }

        # change kind to 'radius', and add radius parameter. k will be ignored
        neighbours = 3 if 'face' in sampling else (5 if level == 0 else 6)
        super(SphereIcosahedron, self).__init__(self.coords, k=neighbours, plotting=plotting, **kwargs)

    def _upward(self, V_ico, F_ico, ind=11):
        V0 = V_ico[ind]
        Z0 = np.array([0, 0, 1])
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R)
        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(F_ico, ind)[0]
        vec = V_ico[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = -np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)
        R2 = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R2)
        return V_ico

    def _find_neighbor(self, F, ind):
        """find a icosahedron neighbor of vertex i"""
        FF = [F[i] for i in range(F.shape[0]) if ind in F[i]]
        FF = np.concatenate(FF)
        FF = np.unique(FF)
        neigh = [f for f in FF if f != ind]
        return neigh

    def _rot_matrix(self, rot_axis, cos_t, sin_t):
        k = rot_axis / np.linalg.norm(rot_axis)
        I = np.eye(3)

        R = []
        for i in range(3):
            v = I[i]
            vr = v*cos_t+np.cross(k, v)*sin_t+k*(k.dot(v))*(1-cos_t)
            R.append(vr)
        R = np.stack(R, axis=-1)
        return R

    def _ico_rot_matrix(self, ind):
        """
        return rotation matrix to perform permutation corresponding to
        moving a certain icosahedron node to the top
        """
        v0_ = self.v0.copy()
        f0_ = self.f0.copy()
        V0 = v0_[ind]
        Z0 = np.array([0, 0, 1])

        # rotate the point to the top (+z)
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        v0_ = v0_.dot(R)

        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(f0_, ind)[0]
        vec = v0_[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)

        R2 = self._rot_matrix(k, ct, st)
        return R.dot(R2)

    def _rotseq(self, V, acc=9):
        """sequence to move an original node on icosahedron to top"""
        seq = []
        for i in range(11):
            Vr = V.dot(self._ico_rot_matrix(i))
            # lexsort
            s1 = np.lexsort(np.round(V.T, acc))
            s2 = np.lexsort(np.round(Vr.T, acc))
            s = s1[np.argsort(s2)]
            seq.append(s)
        return tuple(seq)
