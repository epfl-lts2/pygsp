# -*- coding: utf-8 -*-

r"""
Module documentation.
"""

import numpy as np
from math import ceil, sqrt, log, exp
from copy import deepcopy
from scipy import sparse
from scipy import io
from pygsp import utils


class Graph(object):
    r"""
    parameters:
        - W: Weights matrix
        - A: Adjacency matrix
        - N: Number of nodes
        - d: Degree vector
        - Ne: Egde number
        - gtype: Graph type
        - directed: If the graph is directed
        - lap_type: Laplacian type
        - L: Laplacian
    """

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
            self.N_init_default = True
            self.N = np.shape(self.W)[0]
        if d:
            self.d = d
        else:
            self.d = self.W.sum()
        if Ne:
            self.Ne = Ne
        else:
            self.Ne = np.zeros((self.N), Float)
        if directed:
            self.directed = directed
        else:
            self.directed = utils.is_directed(self)
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


class RandomRegular(Graph):

    def __init__(self, k=None, **kwargs):
        super(RandomRegular, self).__init__(**kwargs)
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
        if self.N_init_default:
            self.N = 64

        # initialization of the param
        try:
            param_nc = param["nc"]
        except (KeyError, TypeError):
            param_nc = 2
        try:
            param_regular = param["regular"]
        except (KeyError, TypeError):
            param_regular = False
        try:
            param_verbose = param["verbose"]
        except (KeyError, TypeError):
            param_verbose = 1
        try:
            param_n_try = param["n_try"]
        except (KeyError, TypeError):
            param_n_try = 50
        try:
            param_distribute = param["distribute"]
        except (KeyError, TypeError):
            param_distribute = False
        try:
            param_connected = param["connected"]
        except (KeyError, TypeError):
            param_connected = True
        try:
            param_set_to_one = param["set_to_one"]
        except (KeyError, TypeError):
            param_set_to_one = False

        if param_connected:
            for x in range(param_n_try):
                W, XCoords, YCoords = create_weight_matrix(self.N, param)

                if gsp_check_connectivity_undirected(W):
                    break
                elif x == param_n_try-1:
                    print("Warning! Graph is not connected")
        else:
            W, XCoords, YCoords = create_weight_matrix(self.N, param)

        if param_set_to_one:
            (x > 0).choose(x, 1)
        self.W = sparse.lil_matrix
        self.W = (self.W + np.transpose(np.conjugate(self.W)))/2
        self.limits = np.array([0, 1, 0, 1])
        self.coords = [XCoords, YCoords]
        if param_regular:
            self.gtype = "regular sensor"
        else:
            self.gtype = "sensor"

        self.directed = False

        def create_weight_matrix(N, param):
            XCoords = np.zeros((N, 1))
            YCoords = np.zeros((N, 1))

            if param_distribute:
                mdim = ceil(sqrt(N))
                for i in np.arange(mdim):
                    for j in np.arange(mdim):
                        if i*mdim + j < N:
                            XCoords[i*mdim + j] = 1/mdim*np.random.rand()+i/mdim
                            YCoords[i*mdim + j] = 1/mdim*np.random.rand()+j/mdim

            # take random coordinates in a 1 by 1 square
            else:
                XCoords = np.random.rand(N, 1)
                YCoords = np.random.rand(N, 1)

            # Compute the distanz between all the points
            target_dist_cutoff = 2*N**(-0.5)
            T = 0.6
            s = sqrt(-target_dist_cutoff**2/(2*log(T)))
            d = gsp_distanz([XCoords, YCoords])
            W = exp(-d**2/(2.*s**2))

            W -= np.diag(np.diag(x))


class Sphere(Graph):

    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        param = kwargs


# Need nothing
class Airfoil(Graph):

    def __init__(self):
        super(Airfoil, self).__init__()
        slef.A = sparse.lil_matrix()

        self.W = (A + np.transpose(np.conjugate(A)))/2

        self.coords = [x, y]


class Bunny(Graph):

    def __init__(self):
        super(Bunny, self).__init__()


class DavidSensorNet(Graph):

    def __init__(self):
        super(DavidSensorNet, self).__init__()


class FullConnected(Graph):

    def __init__(self):
        super(FullConnected, self).__init__()
        if self.N_init_default:
            self.N = 10

        self.W = np.ones((self.N, self.N))-np.identity(self.N)

        tmp = np.arange(0, N).reshape(N, 1)
        self.coords = np.concatenate((np.cos(tpm*2*np.pi/self.N),
                                      np.sin(tpm*2*np.pi/self.N)),
                                     axis=1)
        self.limits = np.array([-1, 1, -1, 1])
        self.gtype = "full"


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
        if self.N_init_default:
            self.N = 16

        inds_i = np.concatenate((np.arange(1, self.N), np.arange(2, self.N+1)),
                                axis=1)
        inds_j = np.concatenate((np.arange(2, self.N+1), np.arange(1, sefl.N)),
                                axis=1)

        np.ones((1, 2*(self.N-1)))
        self.W = sparse.lil_matrix()

        self.coord = np.concatenate((np.arange(1, self.N + 1).reshape(self.N, 1),
                                     np.zeros((1, self.N))),
                                    axis=1)

        self.limits = np.array([0, N+1, -1, 1])

        self.gtype = "path"


class RandomRing(Graph):

    def __init__(self):
        super(RandomRing, self).__init__()
        if self.N_init_default:
            self.N = 64

        position = np.sort(np.random.rand(x))
        position = np.sort(np.random.rand(x, 1), axis=0)

        weight = self.N*np.diff(x, axis=0)
        weightend = self.N*(1 + position[0] - position[-1])

        inds_j = np.conjugate(np.arange(2, self.N + 1).reshape(self.N-1, 1))
        inds_i = np.conjugate(np.arange(1, self.N).reshape(self.N-1, 1))

        # todo
        self.W = sparse.lil_matrix(inds_i, inds_j, weight, N, N)
        self.W(10, 0) = weightend
        self.W += np.conjugate(np.transpose(self.W))

        self.coords = np.concatenate((np.cos(position*2*np.pi),
                                      np.sin(position*2*np.pi)),
                                     axis=1)

        self.limits = np.array([-1, 1, -1, 1])

        self.gtype = 'random-ring'


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
