# -*- coding: utf-8 -*-

r"""
Module documentation.
"""

import numpy as np
from copy import deepcopy
from scipy import sparse
from scipy import io
from pygsp import utils


class Graph(object):

    # All the paramters that needs calculation to be set
    # or not needed are set to None
    def __init__(self, W=None, A=None, N=None, d=None, Ne=None,
                 gtype='unknown', directed=None,
                 lap_type='combinatorial', L=None, **kwargs):

        self.gtype = gtype
        self.lap_type = lap_type

        if W:
            self.W = sparse.lil_matrix(W)
        else:
            self.W = sparse.lil_matrix(0)
        if A:
            self.A = A
        else:
            self.A = sparse.lil_matrix(W > 0)
        if N:
            self.N = N
        else:
            self.N = np.shape(G.W)[0]
        if d:
            self.d = d
        else:
            self.d = self.W.sum()
        if Ne:
            self.Ne = Ne
        else:
            self.Ne = np.zeros((G.N), Float)
        if directed:
            self.directed = directed
        else:
            G.directed = utils.is_directed(self)
            pass
        if L:
            self.L = L
        else:
            self.L = utils.create_laplacian(self)

    def copy_graph_attr(self, gtype, Gn):
        r"""
        TODO write doc
        """
        return deepcopy(self)

    def separate_graph(self):
        r"""
        TODO write func & doc
        """
        raise NotImplementedError("Not implemented yet")

    def subgraph(self, c):
        r"""
        TODO better doc
        This function create a subgraph from G, keeping only the node(s) in c
        """

        sub_G = self
        sub_G.W = [c, c]
        try:
            sub_G.N = len(c)
        except TypeError:
            sub_G.N = 1

        sub_G.gtype = "sub-" + self.gtype

        return sub_G


# Need M
class Grid2d(Graph):

    def __init__(self, M=None, **kwargs):
        super(Grid2d, self).__init__(**kwargs)
        if M:
            self.M = M
        else:
            self.M = self.N

        self.gtype = '2d-grid'
        self.N = self.N * self.M

        # Create weighted adjacency matrix
        K = 2 * self.N - 1
        J = 2 * self.M - 1
        i_inds = np.zeros((K*self.M + J*self.N, 1), dtype=float)
        j_inds = np.zeros((K*self.M + J*self.N, 1), dtype=float)
        for i in xrange(1, self.M):
            i_inds[(i-1) * K + np.arange(0, K)] = (i - 1) * self.N + np.append(range(0, self.N - 1), range(1, self.N))
            j_inds[(i-1) * K + np.arange(0, K)] = (i - 1) * self.N + np.append(range(1, self.N), range(0, self.N - 1))

        for i in xrange(1, self.M - 1):
            i_inds[(K*self.M) + (i-1)*2*self.N + np.arange(1, 2*self.N)] = np.append((i-1)*self.N + np.array(range(1, self.N)), (i*self.N) + np.array(range(1, self.N)))
            j_inds[(K*self.M) + (i-1)*2*self.N + np.arange(1, 2*self.N)] = np.append((i*self.N) + np.array(range(1, self.N)), (i-1)*self.N + np.array(range(1, self.N)))

        self.W = sparse.lil_matrix((self.M * self.N, self.M * self.N))
        # for i_inds, j_inds in
        self.W = sparse.lil_matrix((np.ones((K*self.M+J*self.N, 1)), (i_inds, j_inds)), shape=(self.M*self.N, self.M*self.N))


class Torus(Graph):

    def __init__(self, M=None, **kwargs):
        super(Torus, self).__init__(**kwargs)
        if M:
            self.M = M
        else:
            self.M = self.N

        self.gtype = 'Torus'
        self.directed = False

        # Create weighted adjancency matrix
        K = 2 * self.N
        J = 2 * self.M
        i_inds = np.zeros((K*self.M + J*self.N, 1), dtype=float)
        j_inds = np.zeros((K*self.M + J*self.N, 1), dtype=float)
        for i in xrange(1, self.M):
            i_inds[(i-1)*K + np.arange(0, K)] = (i-1)*self.N + np.append(self.N, np.append(range(0, self.N-1), range(0, self.N)))
            j_inds[(i-1)*K + np.arange(0, K)] = (i-1)*self.N + np.append(range(0, self.N), np.append(self.N, range(0, self.N-1)))
        for i in xrange(1, self.M - 1):
            i_inds[(K*self.M) + (i-1)*2*self.N + np.arange(1, 2*self.N)] = np.append((i-1)*self.N + np.arange(1, self.N), (i*self.N) + np.arange(1, self.N))
            j_inds[(K*self.M) + (i-1)*2*self.N + np.arange(1, 2*self.N)] = np.append((i*self.N) + np.arange(1, self.N), (i-1)*self.N + np.arange(1, self.N))
        i_inds[K*self.M + (self.M-1)*2*self.N + np.arrange(0, 2*self.N)] = np.array([np.arange(0, self.N), (self.M-1)*self.N + np.arange(0, self.N)])
        j_inds[K*self.M + (self.M-1)*2*self.N + np.arrange(0, 2*self.N)] = np.array([(self.M-1)*self.N + np.arange(0, self.N), np.arange(0, self.N)])

        self.W = sparse.lil_matrix((self.M * self.N, self.M * self.N))
        # for i_inds, j_inds in
        self.W = sparse.lil_matrix((np.ones((K*self.M+J*self.N, 1)), (i_inds, j_inds)), shape=(self.M*self.N, self.M*self.N))

        # TODO implementate plot attribute


# Need K
class Comet(Graph):

    def __init__(self, k=None, **kwargs):
        super(Comet, self).__init__(**kwargs)
        if k:
            self.k = k
        else:
            self.k = 12


class LowStretchTree(Graph):

    def __init__(self, k=None, **kwargs):
        super(LowStretchTree, self).__init__(**kwargs)
        if k:
            self.k = k
        else:
            self.k = 6


class RadomRegular(Graph):

    def __init__(self, k=None, **kwargs):
        super(RadomRegular, self).__init__(**kwargs)
        if k:
            self.k = k
        else:
            self.k = 6


class Ring(Graph):

    def __init__(self, k=None, **kwargs):
        super(Ring, self).__init__(**kwargs)
        if k:
            self.k = k
        else:
            self.k = 1


# Need params
class Community(Graph):

    def __init__(self, **kwargs):
        super(Community, self).__init__(**kwargs)
        param = kwargs


class Cube(Graph):

    def __init__(self, **kwargs):
        super(Cube, self).__init__(**kwargs)
        param = kwargs


class Sensor(Graph):

    def __init__(self, **kwargs):
        super(Sensor, self).__init__(**kwargs)
        param = kwargs


class Sphere(Graph):

    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        param = kwargs


# Need nothing
class Airfoil(Graph):

    def __init__(self):
        super(Airfoil, self).__init__()


class Bunny(Graph):

    def __init__(self):
        super(Bunny, self).__init__()


class DavidSensorNet(Graph):

    def __init__(self):
        super(DavidSensorNet, self).__init__()


class FullConnected(Graph):

    def __init__(self):
        super(FullConnected, self).__init__()


class Logo(Graph):

    def __init__(self):
        super(Logo, self).__init__()

        mat = io.loadmat('misc/logogsp.mat')
        self.W = mat['W']
        self.gtype = 'from MAT-file'
        # TODO implementate plot attribute


class Path(Graph):

    def __init__(self):
        super(Path, self).__init__()


class RandomRing(Graph):

    def __init__(self):
        super(RandomRing, self).__init__()


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a : int
        Description.
    b : array_like
        Description.
    c : bool
        Description.

    Returns
    -------
    d : ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.module1.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
