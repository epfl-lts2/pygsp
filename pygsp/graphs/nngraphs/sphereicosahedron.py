# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5



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
    SphereHealpix, SphereEquiangular

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
    # TODO create a new class for 'face' as it is the dual of icosahedron and the dodecahedron
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

        for i in range(level):
            self.divide()
            self.normalize()

        if sampling=='face':
            self.coords = self.coords[self.faces].mean(axis=1)

        self.lat, self.long = self.xyz2latlong()

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

    def divide(self):
        """
        Subdivide a mesh into smaller triangles.
        """
        faces = self.faces
        vertices = self.coords
        face_index = np.arange(len(faces))

        # the (c,3) int set of vertex indices
        faces = faces[face_index]
        # the (c, 3, 3) float set of points in the triangles
        triangles = vertices[faces]
        # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
        src_idx = np.vstack([faces[:, g] for g in [[0, 1], [1, 2], [2, 0]]])
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = self._unique_rows(mid)

        mid = mid[unique]
        src_idx = src_idx[unique]
        mid_idx = inverse[mid_idx] + len(vertices)
        # the new faces, with correct winding
        f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                             mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                             mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                             mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
        # add the 3 new faces per old face
        new_faces = np.vstack((faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((vertices, mid))
        # source ids
        nv = vertices.shape[0]
        identity_map = np.stack((np.arange(nv), np.arange(nv)), axis=1)
        src_id = np.concatenate((identity_map, src_idx), axis=0)

        self.coords = new_vertices
        self.faces = new_faces
        self.intp = src_id

    def normalize(self, radius=1):
        '''
        Reproject to spherical surface
        '''
        vectors = self.coords
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.coords += unit * offset.reshape((-1, 1))

    def xyz2latlong(self):
        x, y, z = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        long = np.arctan2(y, x) + np.pi
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, long

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

    def _unique_rows(self, data, digits=None):
        """
        Returns indices of unique rows. It will return the
        first occurrence of a row that is duplicated:
        [[1,2], [3,4], [1,2]] will return [0,1]
        Parameters
        ---------
        data: (n,m) set of floating point data
        digits: how many digits to consider for the purposes of uniqueness
        Returns
        --------
        unique:  (j) array, index in data which is a unique row
        inverse: (n) length array to reconstruct original
                     example: unique[inverse] == data
        """
        hashes = self._hashable_rows(data, digits=digits)
        garbage, unique, inverse = np.unique(hashes,
                                             return_index=True,
                                             return_inverse=True)
        return unique, inverse

    def _hashable_rows(self, data, digits=None):
        """
        We turn our array into integers based on the precision
        given by digits and then put them in a hashable format.
        Parameters
        ---------
        data:    (n,m) input array
        digits:  how many digits to add to hash, if data is floating point
                 If none, TOL_MERGE will be turned into a digit count and used.
        Returns
        ---------
        hashable:  (n) length array of custom data which can be sorted
                    or used as hash keys
        """
        # if there is no data return immediatly
        if len(data) == 0:
            return np.array([])

        # get array as integer to precision we care about
        as_int = self._float_to_int(data, digits=digits)

        # if it is flat integers already, return
        if len(as_int.shape) == 1:
            return as_int

        # if array is 2D and smallish, we can try bitbanging
        # this is signifigantly faster than the custom dtype
        if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
            # time for some righteous bitbanging
            # can we pack the whole row into a single 64 bit integer
            precision = int(np.floor(64 / as_int.shape[1]))
            # if the max value is less than precision we can do this
            if np.abs(as_int).max() < 2**(precision - 1):
                # the resulting package
                hashable = np.zeros(len(as_int), dtype=np.int64)
                # loop through each column and bitwise xor to combine
                # make sure as_int is int64 otherwise bit offset won't work
                for offset, column in enumerate(as_int.astype(np.int64).T):
                    # will modify hashable in place
                    np.bitwise_xor(hashable,
                                   column << (offset * precision),
                                   out=hashable)
                return hashable

        # reshape array into magical data type that is weird but hashable
        dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
        # make sure result is contiguous and flat
        hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
        return hashable

    def _float_to_int(self, data, digits=None, dtype=np.int32):
        """
        Given a numpy array of float/bool/int, return as integers.
        Parameters
        -------------
        data:   (n, d) float, int, or bool data
        digits: float/int precision for float conversion
        dtype:  numpy dtype for result
        Returns
        -------------
        as_int: data, as integers
        """
        # convert to any numpy array
        data = np.asanyarray(data)

        # if data is already an integer or boolean we're done
        # if the data is empty we are also done
        if data.dtype.kind in 'ib' or data.size == 0:
            return data.astype(dtype)

        # populate digits from kwargs
        if digits is None:
            digits = self._decimal_to_digits(1e-8)
        elif isinstance(digits, float) or isinstance(digits, np.float):
            digits = self._decimal_to_digits(digits)
        elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
            # log.warn('Digits were passed as %s!', digits.__class__.__name__)
            raise ValueError('Digits must be None, int, or float!')

        # data is float so convert to large integers
        data_max = np.abs(data).max() * 10**digits
        # ignore passed dtype if we have something large
        dtype = [np.int32, np.int64][int(data_max > 2**31)]
        # multiply by requested power of ten
        # then subtract small epsilon to avoid "go either way" rounding
        # then do the rounding and convert to integer
        as_int = np.round((data * 10 ** digits) - 1e-6).astype(dtype)

        return as_int


    def _decimal_to_digits(self, decimal, min_digits=None):
        """
        Return the number of digits to the first nonzero decimal.
        Parameters
        -----------
        decimal:    float
        min_digits: int, minumum number of digits to return
        Returns
        -----------
        digits: int, number of digits to the first nonzero decimal
        """
        digits = abs(int(np.log10(decimal)))
        if min_digits is not None:
            digits = np.clip(digits, min_digits, 20)
        return digits
