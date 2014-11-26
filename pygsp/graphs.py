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


class NNGraph(Graph):
    r"""
    Creates a graph from a pointcloud
    parameters:
        - Xin : Input Points
    """

    def __init__(self, Xin, gtype='knn', use_flann=0, center=1, rescale=1, k=10, sigma=0.1, epsilon=0.01 **kwargs):
        super(NNGraph, self).__init__(**kwargs)
        N, d = np.shape(Xin)
        Xout = Xin
        if self.center:
            # TODO Check if equi to repmat
            Xout = Xin - np.kron(Xin.mean(), N)

        if self.rescale:
            # TODO find equivalent to norm
            bounding_radius = 0.5 * norm(max(Xout) - min(Xout))
            # TODO scale =
            Xout *= scale * bounding_radius


class Bunny(NNGraph):

    def __init__(self):
        self.type = "radius"
        self.rescale = 1
        self.center = 1
        self.epsilon = 0.2
        # TODO do the ritgh way when point cloud is merged
        self.Xin = Pointcloud(name="bunny").P
        super(Bunny, self).__init__(Xin, **kwargs)


class Sphere(NNGraph):

    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)


class Cube(NNGraph):

    def __init__(self, **kwargs):
        super(Cube, self).__init__(**kwargs)
        param = kwargs

        try:
            param_raduis = param["radius"]
        except (KeyError, TypeError):
            param_raduis = 1
        try:
            param_nb_pts = param["nb_pts"]
        except (KeyError, TypeError):
            param_nb_pts = 300
        try:
            param_nb_dim = param["nb_dim"]
        except (KeyError, TypeError):
            param_nb_dim = 3
        try:
            param_sampling = param["sampling"]
        except (KeyError, TypeError):
            param_sampling = "random"

        if param_nb_dim > 3:
            raise ValueError("Dimension > 3 not supported yet !")

        if param_sampling == "random":
            if param_nb_dim == 2:
                pts = np.random.rand(param_nb_dim, param_nb_dim)

            if param_nb_dim == 3:
                n = floor(param_nb_dim/6)

                pts = np.zeros((n*6, 3))
                pts[:n, 1:] = np.random.rand(n, 2)
                pts[n:2*n, :] = np.concatenate((np.ones((n, 1)),
                                                np.random.rand(n, 2)),
                                               axis=1)

                pts[2*n:3*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.zeros((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)
                pts[3*n:4*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.ones((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)

                pts[4*n:5*n, :2] = np.random.rand(n, 2)
                pts[5*n:6*n, :] = np.concatenate((np.random.rand(n, 2),
                                                  np.ones((n, 1))),
                                                 axis=1)

        else:
            raise ValueError("Unknown sampling !")

        self.gtype = "knn"
        self.k = 10

        # call of the pcl_graph class
        pclnngraph(pts, param)


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

    def __init__(self, N=256, Nc=None, com_sizes=[], min_com=None, min_deg=None, verboes=1, size_ratio=1, world_density=None **kwargs):
        super(Community, self).__init__(**kwargs)
        param = kwargs

        # Initialisation of the parameters
        self.N = N

        if Nc:
            self.Nc = Nc
        else:

            self.Nc = round(sqrt(self.N))
        if len(com_sizes) != 0:
            if np.sum(com_sizes) != self.N:
                raise ValueError("GSP_COMMUNITY: The sum of the community sizes has to be equal to N")
            else:
                self.com_sizes = com_sizes

        if min_com:
            self.min_com = min_com
        else:
            self.min_com = round(self.N / self.Nc / 3.)

        if min_deg:
            self.min_deg = min_deg
        else:
            self.min_deg = round(self.min_com/2.)

        self.verbose = verbose

        self.size_ratio = size_ratio

        if world_density:
            self.world_density = world_density
        else:
            self.world_density = 1./self.N

        # Begining
        if len(self.com_sizes) == 0:
            com_lims = np.sort(np.random.choice(self.N - (self.min_com - 1)*self.Nc - 1, sefl.Nc - 1) + 1)
            com_lims += np.cumsum((self.min_com-1)*np.ones(np.shape(com_lims)))
            com_lims = np.concatenate((np.array([0]), com_lims, np.array([self.N])))
            self.com_sizes = np.diff(com_lims)

        if self.verbose > 2:
            # TODO
                X = np.zeros((10000, self.Nc + 1))
                for i in np.arange(10000):
                    com_lims_tmp = np.sort(np.random.choice(self.N - (self.min_com - 1)*self.Nc - 1, sefl.Nc - 1) + 1)
                    com_lims_tmp += np.cumsum((self.min_com-1)*np.ones(np.shape(com_lims)))
                    X[i, :] = np.concatenate((np.array([0]), com_lims_tmp, np.array([self.N])))
                dX = np.transpose(np.diff(np.transpose(X)))
                for i in np.arange(self.Nc):


class Cube(NNGraph):

    def __init__(self, radius=1, nb_pts=300, nb_dim=300, sampling="random", **kwargs):
        super(Cube, self).__init__(**kwargs)
        param = kwargs
        self.raduis = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.nb_dim > 3:
            raise ValueError("Dimension > 3 not supported yet !")

        if self.sampling == "random":
            if self.nb_dim == 2:
                pts = np.random.rand(self.nb_dim, self.nb_dim)

            if self.nb_dim == 3:
                n = floor(self.nb_dim/6)

                pts = np.zeros((n*6, 3))
                pts[:n, 1:] = np.random.rand(n, 2)
                pts[n:2*n, :] = np.concatenate((np.ones((n, 1)),
                                                np.random.rand(n, 2)),
                                               axis=1)

                pts[2*n:3*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.zeros((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)
                pts[3*n:4*n, :] = np.concatenate((np.random.rand(n, 1),
                                                  np.ones((n, 1)),
                                                  np.random.rand(n, 1)),
                                                 axis=1)

                pts[4*n:5*n, :2] = np.random.rand(n, 2)
                pts[5*n:6*n, :] = np.concatenate((np.random.rand(n, 2),
                                                  np.ones((n, 1))),
                                                 axis=1)

        else:
            raise ValueError("Unknown sampling !")

        self.gtype = "knn"
        self.k = 10

        # call of the pcl_graph class
        pclnngraph(pts, param)


class Sensor(Graph):

    def __init__(self, N=64, nc=2, regular=False, verbose=1, n_try=50, distribute=False, connected=True, set_to_one=False, **kwargs):
        super(Sensor, self).__init__(**kwargs)
        param = kwargs
        self.N = N
        self.nc = nc
        self.regular = regular
        self.verbose = verbose
        self.n_try = n_try
        self.distribute = distribute
        self.connected = connected
        self.set_to_one = set_to_one

        if self.connected:
            for x in range(self.n_try):
                W, XCoords, YCoords = create_weight_matrix(self.N, self.distribute, self.regular, self.nc)

                if gsp_check_connectivity_undirected(W):
                    break
                elif x == self.n_try-1:
                    print("Warning! Graph is not connected")
        else:
            W, XCoords, YCoords = create_weight_matrix(self.N, self.distribute, self.regular, self.nc)

        if self.set_to_one:
            np.where(x > 0, 1, x)

        # TODO
        self.W = sparse.lil_matrix
        self.W = (self.W + np.transpose(np.conjugate(self.W)))/2
        self.limits = np.array([0, 1, 0, 1])
        self.coords = [XCoords, YCoords]
        if self.regular:
            self.gtype = "regular sensor"
        else:
            self.gtype = "sensor"

        self.directed = False

        def create_weight_matrix(N, param_distribute, param_regular, param_nc):
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

            # TODO gsp_distanz
            d = gsp_distanz([XCoords, YCoords])
            W = exp(-d**2/(2.*s**2))

            W -= np.diag(np.diag(x))

            if param_regular:
                W = get_nc_connection(W, param_nc)
            else:
                W2 = get_nc_connection(W, param_nc)
                np.where(W < T, 0, W)
                np.where(W2 > 0, W2, W)

            return W, XCoords, YCoords

        def get_nc_connection(W, param_nc):
            Wtmp = W
            W = np.zeros(np.shape(W))
            for i in np.arange(np.shape(W)[0]):
                l = Wtemp[i]
                for j in np.arange(param_nc):
                    val = np.max(l)
                    ind = np.argmax(l)
                    W[i, ind] = val
                    l[ind] = 0

            W = (W + np.transpose(np.conjugate(W)))/2.


# Need nothing
class Airfoil(Graph):

    def __init__(self):
        super(Airfoil, self).__init__()
        slef.A = sparse.lil_matrix()

        self.W = (A + np.transpose(np.conjugate(A)))/2

        self.coords = [x, y]


class DavidSensorNet(Graph):

    def __init__(self):
        super(DavidSensorNet, self).__init__()


class FullConnected(Graph):

    def __init__(self, N=10):
        super(FullConnected, self).__init__()
        self.N N

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

    def __init__(self, N=16):
        super(Path, self).__init__()
        self.N = N
        inds_i = np.concatenate((np.arange(1, self.N), np.arange(2, self.N+1)),
                                axis=1)
        inds_j = np.concatenate((np.arange(2, self.N+1), np.arange(1, sefl.N)),
                                axis=1)

        np.ones((1, 2*(self.N-1)))

        # TODO
        self.W = sparse.lil_matrix()

        self.coord = np.concatenate((np.arange(1, self.N + 1).reshape(self.N, 1),
                                     np.zeros((1, self.N))),
                                    axis=1)

        self.limits = np.array([0, N+1, -1, 1])

        self.gtype = "path"


class RandomRing(Graph):

    def __init__(self, N=64):
        super(RandomRing, self).__init__()
        self.N = N

        position = np.sort(np.random.rand(x))
        position = np.sort(np.random.rand(x, 1), axis=0)

        weight = self.N*np.diff(x, axis=0)
        weightend = self.N*(1 + position[0] - position[-1])

        inds_j = np.conjugate(np.arange(2, self.N + 1).reshape(self.N-1, 1))
        inds_i = np.conjugate(np.arange(1, self.N).reshape(self.N-1, 1))

        # TODO
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
