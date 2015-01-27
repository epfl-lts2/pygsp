# -*- coding: utf-8 -*-

r"""
Module documentation.
"""

import os
import os.path
import numpy as np
import random as rd
from math import ceil, sqrt, log, exp, floor
from copy import deepcopy
from scipy import sparse
from scipy import io
from scipy import spatial

from pygsp import utils


class Graph(object):
    r"""
<<<<<<< HEAD
    The main graph object

    It is used to initialize by default every missing field of the subgraphs
    It can also be used alone to initialize customs graphs

    Parameters
    ----------
    W: weights matrix
        default is empty
    A: adjancency matrix
        default is constructed with W
    N: number of nodes
        default is the lenght of the first dimension of W
    d: degree vector
        default
    Ne: edge number
    gtype: graph type
        default is "unknown"
    directed: whether the graph is directed
        default depending of the previous values
    lap_type: laplacian type
        default is "combinatorial"
    L: laplacian
    coords: Coordinates of the vertices
        default is np.array([0, 0])
    plotting: dictionnary conataining the plotting parameters

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Graph()
=======
    parameters:
        - W: Weights matrix
        - A: Adjacency matrix
        - N: Number of nodes
        - d: Degree vector
        - Ne: Edges number
        - gtype: Graph type
        - directed: If the graph is directed
        - lap_type: Laplacian type
        - L: Laplacian
        - plotting: dictionnary containing the plotting parameters
>>>>>>> default_graphs
    """

    # All the parameters that needs calculation to be set
    # or not needed are set to None
    def __init__(self, W=None, A=None, N=None, d=None, Ne=None,
                 gtype='unknown', directed=None, coords=None,
                 lap_type='combinatorial', L=None, plotting=None, **kwargs):

        self.gtype = gtype
        self.lap_type = lap_type

        if W is not None:
            self.W = sparse.lil_matrix(W)
        else:
            self.W = sparse.lil_matrix(0)
        if A is not None:
            self.A = A
        else:
            self.A = sparse.lil_matrix(W > 0)
        if N is not None:
            self.N = N
        else:
            self.N = np.shape(self.W)[0]
        if d is not None:
            self.d = d
        else:
            self.d = self.W.sum()
        if Ne is not None:
            self.Ne = Ne
        else:
            self.Ne = self.W.nnz
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.zeros((self.N, 2))
        if directed:
            self.directed = directed
        else:
            self.directed = utils.is_directed(self)
            pass
        if L is not None:
            self.L = L

        else:
            self.L = utils.create_laplacian(self)

        # Plotting default parameters
        self.plotting = {}
        if 'edge_width' in plotting:
            self.plotting['edge_width'] = plotting['edge_width']
        else:
            self.plotting['edge_width'] = 1
        if 'edge_color' in plotting:
            self.plotting['edge_color'] = plotting['edge_color']
        else:
            self.plotting['edge_color'] = np.array([255, 88, 41])/255
        if 'edge_style' in plotting:
            self.plotting['edge_style'] = plotting['edge_style']
        else:
            self.plotting['edge_style'] = '-'
        if 'vertex_size' in plotting:
            self.plotting['vertex_size'] = plotting['vertex_size']
        else:
            self.plotting['vertex_size'] = 50
        if 'vertex_color' in plotting:
            self.plotting['vertex_color'] = plotting['vertex_color']
        else:
            self.plotting['vertex_color'] = 'b'

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

    Parameters:
        Xin: Input Points
        use_flann: Whether flann method should be used (knn is otherwise used)
            default is False
            (not implemented yet)
        center: center the data
            default is True
        rescale: rescale the data (in a 1-ball)
            default is True
        k: number of neighbors for knn
            default is 10
        sigma: variance of the distance kernel
            default is 0.1
        epsilon: radius for the range search
            default is 0.01
        gtype: the type of graph
            default is "knn"

    Examples
    --------
    >>> from pygsp import graphs
    >>> Xin = np.arange(16)
    >>> G = graphs.NNGraph(Xin)
    """

    def __init__(self, Xin, NNtype='knn', use_flann=False, center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01, gtype=None,
                 plotting=None, symetrize_type='average', **kwargs):
        if Xin is None:
            raise ValueError("You must enter a Xin to process the NNgraph")
        else:
            self.Xin = Xin
        self.NNtype = NNtype
        self.use_flann = use_flann
        self.center = center
        self.rescale = rescale
        self.k = k
        self.sigma = sigma
        self.epsilon = epsilon
        if gtype is None:
            self.gtype = "nearest neighbors"
        else:
            self.gtype = gtype + ", NNgraph"
        if plotting:
            self.plotting = plotting
        else:
            self.plotting = {}
        self.symetrize_type = symetrize_type
        param = kwargs

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if self.center:
            Xout = self.Xin - np.kron(np.ones((N, 1)),
                                      np.mean(self.Xin, axis=0))

        if self.rescale:
            bounding_radius = 0.5*np.linalg.norm(np.amax(Xout, axis=0)
                                                 - np.amin(Xout, axis=0), 2)
            scale = np.power(N, 1./float(min(d, 3)))/10.
            Xout *= scale/bounding_radius

        if self.NNtype == "knn":
            spi = np.zeros((N*self.k))
            spj = np.zeros((N*self.k))
            spv = np.zeros((N*self.k))

            # since we didn't find a good flann python library yet,
            # we wont implement it for now
            if self.use_flann:
                raise NotImplementedError("Suitable library for flann has not \
                                          been found yet.")
            else:
                kdt = spatial.KDTree(Xout)
                D, NN = kdt.query(Xout, k=k + 1)

            for i in range(N):
                spi[i*k:(i+1)*k] = np.kron(np.ones((k)), i)
                spj[i*k:(i+1)*k] = NN[i, 1:]
                spv[i*k:(i+1)*k] = np.exp(-np.power(D[i, 1:], 2)/float(self.sigma))

            self.W = sparse.csc_matrix((spv, (spi, spj)),
                                       shape=(np.shape(self.Xin)[0],
                                              np.shape(self.Xin)[0]))

        elif self.NNtype == "radius":

            kdt = spatial.KDTree(Xout)
            D, NN = kdt.query(Xout, k=None, distance_upper_bound=epsilon)
            count = 0
            for i in range(N):
                count = count + len(NN[i])

            spi = np.zeros((count))
            spj = np.zeros((count))
            spv = np.zeros((count))

            start = 0
            for i in range(N):
                leng = len(NN[i]) - 1
                spi[start:start+leng] = np.kron(np.ones((leng)), i)
                spj[start:start+leng] = NN[i][1:]
                spv[start:start+leng] = np.exp(-np.power(D[i][1:], 2)/float(self.sigma))
                start = start+leng

            W = sparse.csc_matrix((spv, (spi, spj)),
                                  shape=(np.shape(self.Xin)[0],
                                         np.shape(self.Xin)[0]))

        else:
            raise ValueError("Unknown type : allowed values are knn, radius")

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError("Weight matrix W is not square")

        # Symetry checks
        if utils.is_directed(w):
            W = utils.symetrize(W, symetrize_type=self.symetrize_type)
        else:
            print('The matrix W is symmetric')

        self.N = N
        self.W = W
        self.coords = Xout

        super(NNGraph, self).__init__(N=self.N, gtype=self.gtype, W=self.W,
                                      plotting=self.plotting,
                                      coords=self.coords, **kwargs)


class Bunny(NNGraph):
    r"""
    Example graph extracted from matlab data

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Bunny()
    """

    def __init__(self, **kwargs):

        self.NNtype = "radius"
        self.rescale = True
        self.center = True
        self.epsilon = 0.2
        self.gtype = "Bunny"

        bunny = PointsCloud("bunny")
        self.Xin = bunny.Xin

        self.plotting = {"vertex_size": 10}

        super(Bunny, self).__init__(Xin=self.Xin, center=self.center,
                                    rescale=self.rescale, epsilon=self.epsilon,
                                    plotting=self.plotting, NNtype=self.NNtype,
                                    **kwargs)


class Cube(NNGraph):
    r"""
    Creates the graph of an hyper-cube

    Parameters
    ----------
    radius: edge lenght
        default is 1
    nb_pts: number of vertices
        default is 300
    nb_dim: dimension
        default is 3
    sampling: variance of the distance kernel
        default is "random"
        (Can now only be 'random')

    Examples
    --------
    >>> from pygsp import graphs
    >>> radius = 5
    >>> G = graphs.Cube(radius=radius)
    """

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling="random",
                 **kwargs):
        param = kwargs
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.nb_dim > 3:
            raise ValueError("Dimension > 3 not supported yet !")

        if self.sampling == "random":
            if nb_dim == 2:
                pts = np.random.rand(self.nb_pts, self.nb_pts)

            elif self.nb_dim == 3:
                n = floor(self.nb_pts/6)

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

        self.NNtype = "radius"
        self.gtype = "Cube"
        self.k = 10

        super(Cube, self).__init__(Xin=pts, NNtype=self.NNtype, gtype=self.gtype, k=self.k, **kwargs)


class Sphere(NNGraph):

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling="random",
                 **kwargs):
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.sampling == "random":
            pts = np.random.normal(0, 1, (self.nb_pts, self.nb_dim))
            for i in xrange(self.nb_pts):
                pts[i] /= np.linalg.norm(pts[i])
        else:
            raise ValueError("Unknow sampling!")

        self.NNtype = "knn"
        self.k = 10
        self.gtype = "Sphere"

        super(Sphere, self).__init__(Xin=pts, NNtype=self.NNtype, k=self.k,
                                     gtype=self.gtype, **kwargs)


class TwoMoons(NNGraph):
    r"""
    Creates a 2 dimensional graph of the Two Moons

    Parameters
    ----------
    moontype: You have the freedom to chose if you want to create a standard two_moons graph or a synthetised one (default is 'standard').
        'standard': create a two_moons graph from a based graph.
            sigmag: variance of the distance kernel
                default is 0.05

        'synthetised': create a synthetised two_moons
            N: Number of vertices
                default is 2000
            sigmad: variance of the data (do not set it to high or you won't see anything)
                default is 0.05
            d: distance of the two moons
                default is 0.5

    Examples
    --------
    >>> from pygsp import graphs
    >>> G1 = graphs.TwoMoons(moontype='standard')
    >>> 
    >>> G2 =  graphs.TwoMoons(moontype='synthetised', N=1000, sigmad=0.1, d=1)
    """


    def __init__(self, moontype="standard", sigmag=0.05, N=400, sigmad=0.07, d=0.5):

        self.k = 5
        self.sigma = sigmag

        if moontype == "standard":
            two_moons = PointsCloud("two_moons")
            self.Xin = two_moons.Xin

            self.gtype = "Two Moons standard"
            self.labels = 2*(np.where(np.arange(1, N+1).reshape(N, 1) > 1000, 1, 0) + 1)

            super(TwoMoons, self).__init__(Xin=self.Xin, sigma=sigmag, labels=self.labels, gtype=self.gtype, k=self.k)

        elif moontype == 'synthetised':
            self.gtype = "Two Moons synthetised"

            N1 = floor(N/2)
            N2 = N - N1

            # Moon 1
            phi1 = np.random.rand(N1, 1)*np.pi
            r1 = 1
            rb = sigmad*np.random.normal(size=(N1, 1))
            ab = np.random.rand(N1, 1)*2*np.pi
            b = rb*np.exp(1j*ab)
            bx = np.real(b)
            by = np.imag(b)

            moon1x = np.cos(phi1)*r1 + bx + 0.5
            moon1y = -np.sin(phi1)*r1 + by - (d-1)/2

            # Moon 2
            phi2 = np.random.rand(N2, 1)*np.pi
            r2 = 1
            rb = sigmad*np.random.normal(size=(N2, 1))
            ab = np.random.rand(N2, 1)*2*np.pi
            b = rb*np.exp(1j*ab)
            bx = np.real(b)
            by = np.imag(b)

            moon2x = np.cos(phi2)*r2 + bx - 0.5
            moon2y = np.sin(phi2)*r2 + by + (d-1)/2

            self.Xin = np.concatenate((np.concatenate((moon1x, moon1y), axis=1), np.concatenate((moon2x, moon2y), axis=1)))
            self.labels = 2*(np.where(np.arange(1, N+1).reshape(N, 1) > N1, 1, 0) + 1)

            super(TwoMoons, self).__init__(Xin=self.Xin, sigma=sigmag, labels=self.labels, gtype=self.gtype, k=self.k)


# Need M
class Grid2d(Graph):
    r"""
    Creates a 2 dimensional grid graph

    Parameters
    ----------
    Nv: Number of vertices along the first dimension
        default is 16
    Mv: Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Grid2d(Nv = 32)
    """

    def __init__(self, Nv=16, Mv=None, **kwargs):
        self.Nv = Nv
        if Mv:
            self.Mv = Mv
        else:
            self.Mv = Nv

        self.gtype = '2d-grid'
        self.N = self.Nv * self.Mv

        # Create weighted adjacency matrix
        K = 2*(self.Nv-1)
        J = 2*(self.Mv-1)

        i_inds = np.zeros((K*self.Mv + J*self.Nv), dtype=float)
        j_inds = np.zeros((K*self.Mv + J*self.Nv), dtype=float)

        for i in xrange(self.Mv):
            i_inds[i*K + np.arange(K)] = i*self.Nv + np.concatenate((np.arange(self.Nv-1), np.arange(1, self.Nv)))
            j_inds[i*K + np.arange(K)] = i*self.Nv + np.concatenate((np.arange(1, self.Nv), np.arange(self.Nv-1)))

        for i in xrange(self.Mv-1):
            i_inds[(K*self.Mv) + i*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate((i*self.Nv + np.arange(self.Nv), (i+1)*self.Nv + np.arange(self.Nv)))
            j_inds[(K*self.Mv) + i*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate(((i+1)*self.Nv + np.arange(self.Nv), i*self.Nv + np.arange(self.Nv)))

        self.W = sparse.csc_matrix((np.ones((K*self.Mv+J*self.Nv)), (i_inds, j_inds)), shape=(self.Mv*self.Nv, self.Mv*self.Nv))

        xtmp = np.kron(np.ones((self.Mv, 1)), (np.arange(self.Nv)/float(self.Nv)).reshape(self.Nv, 1))
        ytmp = np.sort(np.kron(np.ones((self.Nv, 1)), np.arange(self.Mv)/float(self.Mv)).reshape(self.Mv*self.Nv, 1), axis=0)
        self.coords = np.concatenate((xtmp, ytmp), axis=1)

        self.plotting = {"limits": np.array([-1/self.Nv, 1 + 1/self.Nv, 1/self.Mv, 1 + 1/self.Mv]),
                         "vertex_size": 30}

        super(Grid2d, self).__init__(N=self.N, W=self.W, gtype=self.gtype, plotting=self.plotting, coords=self.coords, **kwargs)


class Torus(Graph):
    r"""
    Creates a Torus graph

    Parameters
    ----------
    Nv: Number of vertices along the first dimension
        default is 16
    Mv: Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> Nv = 32
    >>> G = graphs.Torus(Nv=Nv)

    """

    def __init__(self, Nv=16, Mv=None, **kwargs):
        self.Nv = Nv
        if Mv:
            self.Mv = Mv
        else:
            self.Mv = Nv

        self.gtype = 'Torus'
        self.directed = False

        # Create weighted adjancency matrix
        K = 2 * self.Nv
        J = 2 * self.Mv
        i_inds = np.zeros((K*self.Mv + J*self.Nv), dtype=float)
        j_inds = np.zeros((K*self.Mv + J*self.Nv), dtype=float)

        for i in xrange(self.Mv):
            i_inds[i*K + np.arange(K)] = i*self.Nv + np.concatenate((np.array([self.Nv-1]), np.arange(self.Nv-1), np.arange(self.Nv)))
            j_inds[i*K + np.arange(K)] = i*self.Nv + np.concatenate((np.arange(self.Nv), np.array([self.Nv-1]), np.arange(self.Nv-1)))

        for i in xrange(self.Mv-1):
            i_inds[K*self.Mv + i*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate((i*self.Nv + np.arange(self.Nv), (i+1)*self.Nv + np.arange(self.Nv)))
            j_inds[K*self.Mv + i*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate(((i+1)*self.Nv + np.arange(self.Nv), i*self.Nv + np.arange(self.Nv)))

        i_inds[K*self.Mv + (self.Mv-1)*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate((np.arange(self.Nv), (self.Mv-1)*self.Nv + np.arange(self.Nv)))
        j_inds[K*self.Mv + (self.Mv-1)*2*self.Nv + np.arange(2*self.Nv)] = np.concatenate(((self.Mv-1)*self.Nv + np.arange(self.Nv), np.arange(self.Nv)))

        self.W = sparse.csc_matrix((np.ones((K*self.Mv+J*self.Nv)), (i_inds, j_inds)), shape=(self.Mv*self.Nv, self.Mv*self.Nv))

        # Create Coordinate
        T = 1.5 + np.sin(np.arange(self.Mv)*2*np.pi/self.Mv).reshape(1, self.Mv)
        U = np.cos(np.arange(self.Mv)*2*np.pi/self.Mv).reshape(1, self.Mv)
        xtmp = np.cos(np.arange(self.Nv).reshape(self.Nv, 1)*2*np.pi/self.Nv)*T
        ytmp = np.sin(np.arange(self.Nv).reshape(self.Nv, 1)*2*np.pi/self.Nv)*T
        ztmp = np.kron(np.ones((self.Nv, 1)), U)
        self.coords = np.concatenate((np.reshape(xtmp, (self.Mv*self.Nv, 1),
                                      order='F'),
                                      np.reshape(ytmp, (self.Mv*self.Nv, 1),
                                      order='F'),
                                      np.reshape(ztmp, (self.Mv*self.Nv, 1),
                                      order='F')),
                                     axis=1)

        self.plotting = {"vertex_size": 30,
                         "limits": np.array([-2.5, 2.5, -2.5, 2.5, -2.5, 2.5])}

        super(Torus, self).__init__(W=self.W, directed=self.directed, gtype=self.gtype, coords=self.coords, plotting=self.plotting, **kwargs)


# Need K
class Comet(Graph):
    r"""
    Creates a Comet graph

    Parameters
    ----------
    Nv: Number of vertices along the first dimension
        default is 16
    Mv: Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Comet() (== graphs.Comet(Nv=32, k=12))

    """

    def __init__(self, Nv=32, k=12, **kwargs):
        self.Nv = Nv
        self.k = k
        self.gtype = 'Comet'

        # Create weighted adjancency matrix
        i_inds = np.concatenate((np.zeros((self.k)), np.arange(self.k)+1,
                                 np.arange(self.k, self.Nv-1),
                                 np.arange(self.k+1, self.Nv)))
        j_inds = np.concatenate((np.arange(self.k)+1, np.zeros((self.k)),
                                 np.arange(self.k+1, self.Nv),
                                 np.arange(self.k, self.Nv-1)))

        self.W = sparse.csc_matrix((np.ones((np.size(i_inds))),
                                    (i_inds, j_inds)),
                                   shape=(self.Nv, self.Nv))

        tmpcoords = np.zeros((self.Nv, 2))
        inds = np.arange(k)+1
        tmpcoords[1:k+1, 0] = np.cos(inds*2*np.pi/k)
        tmpcoords[1:k+1, 1] = np.sin(inds*2*np.pi/k)
        tmpcoords[k+1:, 0] = np.arange(1, self.Nv-k)+1
        self.coords = tmpcoords

        self.plotting = {"limits": np.array([-2, np.max(tmpcoords[:, 0]),
                                             np.min(tmpcoords[:, 1]),
                                             np.max(tmpcoords[:, 1])])}

        super(Comet, self).__init__(W=self.W, coords=self.coords,
                                    plotting=self.plotting,
                                    gtype=self.gtype, **kwargs)


class LowStretchTree(Graph):
    r"""
    Creates a low stretch tree graph

    Parameters
    ----------
    k: 2^k points on each side of the grid of vertices
        default 6
    """

    def __init__(self, k=6, **kwargs):

        start_nodes = np.array([1, 1, 3])
        end_nodes = np.array([2, 3, 4])

        W = sparse.csc_matrix((np.ones((3)), (start_nodes, end_nodes)),
                       shape=(4, 4))
        W = W + W.getH()

        XCoords = np.array([1, 2, 1, 2])
        YCoords = np.array([1, 1, 2, 2])

        for p in xrange(2, k+1):
            # TODO the ii/jj part

            YCoords = np.kron(np.ones((1, 2)), YCoords)
            YCoords_new = np.array([YCoords, YCoords+2**(p-1)])
            YCoords = YCoords_new
            XCoords_new = np.array([XCoords, XCoords+2**(p-1)])
            XCoords = np.kron(np.ones((1, 2)), XCoords_new)

        self.coords = np.array([np.transpose(XCoords), np.transpose(YCoords)])
        self.limits = np.array([0, 2**k+1, 0, 2**k+1])
        self.N = (2**k)**2
        self.W = W
        self.root = 4**(k-1)
        self.gtype = "low strech tree"

        self.plotting = {"edges_width": 1.25,
                         "vertex_sizee": 75}

        super(LowStretchTree, self).__init__(W=self.W, coords=self.coords,
                                             N=self.N, limits=self.limits,
                                             root=self.root, gtype=self.gtype,
                                             plotting=self.plotting, **kwargs)


class RandomRegular(Graph):
    r"""
    Creates a random regular graph

    Parameters
    ----------
    N: Number of nodes
        default is 64
    k: Number of connections of each nodes
        default is 6
    """

    def __init__(self, N=64, k=6, **kwargs):
        self.N = N
        self.k = k

        self.gtype = "random_regular"
        self.W = createRandRegGraph(self.N. self.k)

        super(RandomRegular, self).__init__(W=self.W, gtype=self.gtype,
                                            **kwargs)

        def createRandRegGraph(vertNum, deg):
            r"""
            createRegularGraph - creates a simple d-regular undirected graph
            simple = without loops or double edges
            d-reglar = each vertex is adjecent to d edges

            input arguments:
                vertNum: number of vertices
                deg: the degree of each vertex

            output arguments:
              A - A sparse matrix representation of the graph

            algorithm :
            "The pairing model": create n*d 'half edges'.
            repeat as long as possible: pick a pair of half edges
              and if it's legal (doesn't creat a loop nor a double edge)
              add it to the graph

            reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.7957&rep=rep1&type=pdf
            """

            n = vertNum
            d = deg
            matIter = 10

            # continue until a proper graph is formed
            if (n*d) % 2 == 1:
                raise ValueError("createRandRegGraph input err:\
                                 n*d must be even!")

            # a list of open half-edges
            U = np.kron(np.ones((1, d)), np.arange(n)+1)

            # the graphs adajency matrix
            A = sparse.csc_matrix(n, n)

            edgesTested = 0
            repetition = 1

            # check that there are no loops nor parallel edges
            while np.size(U) != 0 and repetition < matIter:
                edgesTested += 1

                # print progess
                if edgesTested % 5000 == 0:
                    print("createRandRegGraph() progress: edges=%d/%d\n" % (edgesTested, n*d))

                # chose at random 2 half edges
                v1 = ceil(rd.random()*np.shape(U)[0])
                i2 = ceil(rd.random()*np.shape(U)[0])
                v1 = U[i1]
                v2 = U[i2]

                # check that there are no loops nor parallel edges
                if vi == v2 or A[v1, v2] == 1:
                    # restart process if needed
                    if edgesTested == n*d:
                        repetition = repetition + 1
                        edgesTested = 0
                        U = np.kron(np.ones((1, d)), np.arange(n)+1)
                        A = sparse.csc_matrix(n, n)
                else:
                    # add edge to graph
                    A[v1, v2] = 1
                    A[v2, v1] = 1

                    # remove used half-edges
                    v = sorted([v1, v2])
                    U = np.concatenate((U[1:v[0]], U[v[0]+1:v[1]], U[v[1]+1:]))

            isRegularGraph(A)

            return A

        def isRegularGraph(G):

            msg = "the grpah G "

            # check symmetry
            tmp = (G-G.getH())
            if np.sum((tmp.getH()*tmp).diagonal()) > 0:
                msg += "is not symetric, "

            # check parallel edged
            if G.max(axis=None) > 1:
                msg += "has parallel edges, "

            # check that d is d-regular
            d_vec = G.sum(axis=0)
            if np.min(d_vec) < d_vec[:, 0] and np.max(d_vec) > d_vec[:, 0]:
                msg += "not d-regular, "

            # check that g doesn't contain any loops
            if G.diagonal().any() > 0:
                msg += "has self loops, "

            else:
                msg += "is ok"

            print(msg)


class Ring(Graph):
    r"""
    Creates a ring graph

    Parameters
    ----------
    N: Number of vertices
        default is 64
    k: Number of neighbors in each directions
        default is 1
    """

    def __init__(self, N=64, k=1, **kwargs):
        self.N = N
        self.k = k

        if self.k > self.N/2:
            raise ValueError("Too many neighbors requested.")

        # Create weighted adjancency matrix
        if self.k == self.N/2:
            num_edges = self.N*(self.k-1) + self.N/2
        else:
            num_edges = self.N*self.k

        i_inds = np.zeros((2*num_edges))
        j_inds = np.zeros((2*num_edges))

        all_inds = np.arange(self.N)
        for i in xrange(min(self.k, floor((self.N-1)/2))):
            i_inds[(i*2*self.N):(i*2*self.N + self.N)] = all_inds
            j_inds[(i*2*self.N):(i*2*self.N + self.N)] = np.remainder(all_inds + i +1, self.N)
            i_inds[(i*2*self.N + self.N):((i + 1)*2*self.N)] = np.remainder(all_inds + i +1, self.N)
            j_inds[(i*2*self.N + self.N):((i + 1)*2*self.N)] = all_inds

        if self.k == self.N/2:
            i_inds[(2*self.N*(self.k - 1)):(2*self.N*(self.k - 1)+self.N)] = all_inds
            i_inds[(2*self.N*(self.k - 1)):(2*self.N*(self.k - 1)+self.N)] = np.remainder(all_inds + k +1, self.N)

        self.W = sparse.csc_matrix((np.ones((2*num_edges)), (i_inds, j_inds)), shape=(self.N, self.N))

        self.coords = np.concatenate((np.cos(np.arange(self.N).reshape(self.N, 1)*2*np.pi/float(self.N)),
                                      np.sin(np.arange(self.N).reshape(self.N, 1)*2*np.pi/float(self.N))),
                                     axis=1)

        self.plotting = {"limits": np.array([-1, 1, -1, 1])}

        if self.k == 1:
            self.gtype = "ring"
        else:
            self.gtype = "k-ring"

        super(Ring, self).__init__(W=self.W, N=self.N, gtype=self.gtype,
                                   coords=self.coords, plotting=self.plotting,
                                   **kwargs)


# Need params
class Community(Graph):
    r"""
    Create a community graph

    Parameters
    ----------
    N: Number of nodes
        default is 256
    Nc: Number of communities
        default is round(sqrt(N)/2)
    com_sizes: Size of the communities
        default is is random
    min_comm: Minimum size of the communities
        default is round(N/Nc/3)
    min_deg: Minimum degree of each node
        default is round(min_comm/2) (not implemented yet)
    verbose: Verbosity output
        default is 1
    size_ratio: Ratio between the radius of world and the radius of communities
        default is 1
    world_density: Probability of a random edge between any pair of edges
        default is 1/N
    """

    def __init__(self, N=256, Nc=None, com_sizes=[], min_com=None,
                 min_deg=None, verbose=1, size_ratio=1, world_density=None,
                 **kwargs):
        param = kwargs

        # Initialisation of the parameters
        self.N = N
        if Nc:
            self.Nc = Nc
        else:
            self.Nc = round(sqrt(self.N))

        if len(com_sizes) != 0:
            if np.sum(com_sizes) != self.N:
                raise ValueError("GSP_COMMUNITY: The sum of the community \
                                 sizes has to be equal to N")
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
            com_lims = np.sort(np.random.choice(self.N - (self.min_com - 1)*self.Nc - 1, self.Nc - 1) + 1)
            com_lims += np.cumsum((self.min_com-1)*np.ones(np.shape(com_lims)))
            com_lims = np.concatenate((np.array([0]), com_lims, np.array([self.N])))
            self.com_sizes = np.diff(com_lims)

        if self.verbose >= 2:
                X = np.zeros((10000, self.Nc + 1))
                # pick randomly param.Nc-1 points to cut the rows in communtities:
                for i in xrange(10000):
                    com_lims_tmp = np.sort(np.random.choice(self.N - (self.min_com - 1)*self.Nc - 1, self.Nc - 1) + 1)
                    com_lims_tmp += np.cumsum((self.min_com-1)*np.ones(np.shape(com_lims)))
                    X[i, :] = np.concatenate((np.array([0]), com_lims_tmp, np.array([self.N])))
                dX = np.transpose(np.diff(np.transpose(X)))
                for i in xrange(self.Nc):
                    # TODO figure; hist(dX(:,i), 100); title('histogram of row community size'); end
                    pass
                del X
                del com_lims_tmp

        rad_world = self.size_ratio*sqrt(self.N)
        com_coords = rad_world*np.concatenate((-np.cos(2*np.pi*(np.arange(self.Nc) + 1).reshape(10, 1)/self.Nc),
                                               np.sin(2*np.pi*(np.arange(self.Nc) + 1).reshape(10, 1)/self.Nc)),
                                              axis=1)

        self.coords = np.ones((self.N, 2))

        # create uniformly random points in the unit disc
        for i in xrange(self.N):
            # use rejection sampling to sample from a unit disc (probability = pi/4)
            while np.linalg.norm(self.coords[i], 2) >= 0.5:
                # sample from the square and reject anything outside the circle
                self.coords[i] = rd.uniform(-0.5, 0.5), rd.uniform(-0.5, 0.5)

        info = {"node_com": np.zeros((self.N, 1))}

        # add the offset for each node depending on which community it belongs to
        for i in xrange(self.Nc):
            com_size = self.com_sizes[i]
            rad_com = sqrt(com_size)
            node_ind = np.arange(com_lims[i] + 1, com_lims[i+1])
            self.coords[node_ind] = rad_com*self.coords[node_ind] + com_coords[i]
            info["node_com"] = i

        D = utils.distanz(np.transpose(self.coords))
        W = exp(-np.power(D, 2))
        W = np.where(W < 1e-3, 0, W)

        # When we make W symetric, the density get bigger (because we add a ramdom number of values)
        density = self.N/(2.-1./self.world_density)

        W = W + np.abs(sparse.rand(self.N, self.N, density=density))
        w = (W + W.getH())/2  # make W symetric

        W = np.where(np.abs(W) > 0, 1, W).astype(float)
        self.W = sparse.coo_matrix(W)
        self.gtype = "Community"

        # return additional info about the communities
        info["com_lims"] = com_lims
        info["com_coords"] = com_coords
        info["com_sizes"] = self.com_sizes
        self.info = info

        super(Community, self).__init__(W=self.W, gtype=self.gtype, coords=self.coords, info=self.info, **kwargs)


class Sensor(Graph):
    r"""
    Creates a random sensor graph

    Parameters
    ----------
    N: Number of nodes
        default is 64
    nc: Minimum number of connections
        default is 1
    regular: Flag to fix the number of connections to nc
        default is False
    verbose: Verbosity parameter
        default is True
    n_try: Number of attempt to create the graph
        default is 50
    distribute: To distribute the points more evenly
        default is False
    connected: To force the graph to be connected
        default is True
    set_to_one:
        default is False
    """

    def __init__(self, N=64, nc=2, regular=False, verbose=True, n_try=50, distribute=False, connected=True, set_to_one=False, **kwargs):
        param = kwargs
        self.N = N
        self.nc = nc
        self.regular = regular
        self.verbose = verbose
        self.n_try = n_try
        self.distribute = distribute
        self.connected = connected
        self.set_to_one = set_to_one

        def create_weight_matrix(N, param_distribute, param_regular, param_nc):
            XCoords = np.zeros((N, 1))
            YCoords = np.zeros((N, 1))

            if param_distribute:
                mdim = int(ceil(sqrt(N)))
                for i in xrange(mdim):
                    for j in xrange(mdim):
                        if i*mdim + j < N:
                            XCoords[i*mdim + j] = np.array(1./float(mdim)*np.random.rand()+i/float(mdim))
                            YCoords[i*mdim + j] = np.array(1./float(mdim)*np.random.rand()+j/float(mdim))

            # take random coordinates in a 1 by 1 square
            else:
                XCoords = np.random.rand(N, 1)
                YCoords = np.random.rand(N, 1)

            # Compute the distanz between all the points
            target_dist_cutoff = 2*N**(-0.5)
            T = 0.6
            s = sqrt(-target_dist_cutoff**2/(2*log(T)))
            d = utils.distanz(x=XCoords, y=YCoords)
            W = np.exp(-d**2/(2.*s**2))
            W -= np.diag(np.diag(W))

            if param_regular:
                W = get_nc_connection(W, param_nc)

            else:
                W2 = get_nc_connection(W, param_nc)
                W = np.where(W < T, 0, W)
                W = np.where(W2 > 0, W2, W)

            return W, XCoords, YCoords

        def get_nc_connection(W, param_nc):
            Wtmp = W
            W = np.zeros(np.shape(W))

            for i in xrange(np.shape(W)[0]):
                l = Wtmp[i]
                for j in xrange(param_nc):
                    val = np.max(l)
                    ind = np.argmax(l)
                    W[i, ind] = val
                    l[ind] = 0

            W = (W + np.transpose(np.conjugate(W)))/2.

            return W

        if self.regular:
            self.gtype = "regular sensor"
        else:
            self.gtype = "sensor"
        self.directed = False

        if self.connected:
            for x in xrange(self.n_try):
                W, XCoords, YCoords = create_weight_matrix(self.N,
                                                           self.distribute,
                                                           self.regular,
                                                           self.nc)

                self.W = W

                if utils.check_connectivity(self):
                    self.W = W
                    break
                elif x == self.n_try-1:
                    print("Warning! Graph is not connected")
        else:
            W, XCoords, YCoords = create_weight_matrix(self.N, self.distribute,
                                                       self.regular, self.nc)

        if self.set_to_one:
            W = np.where(W > 0, 1, W)

        W = sparse.lil_matrix(W)
        self.W = (W + W.getH())/2
        self.coords = np.concatenate((XCoords, YCoords), axis=1)

        self.plotting = {"limits": np.array([0, 1, 0, 1])}

        super(Sensor, self).__init__(W=self.W, N=self.N, gtype=self.gtype, coords=self.coords, plotting=self.plotting, directed=self.directed, **kwargs)


# Need nothing
class Airfoil(Graph):
    r"""
    Creates the aifoil graph
    """

    def __init__(self):

        airfoil = PointsCloud("airfoil")
        i_inds = airfoil.i_inds
        j_inds = airfoil.j_inds

        A = sparse.coo_matrix((np.ones((12289)), (np.reshape(i_inds-1, (12289)), np.reshape(j_inds-1, (12289)))), shape=(4253, 4253))
        self.W = (A + sparse.coo_matrix.getH(A))/2

        x = airfoil.x
        y = airfoil.y

        coords = np.array([x, y])
        self.coords = coords.reshape(2, 4253).transpose()
        self.gtype = 'Airfoil'

        self.plotting = {"limits": np.array([-1e-4, 1.01*np.max(x), -1e-4, 1.01*np.max(y)]),
                         "vertex_size": 30}

        super(Airfoil, self).__init__(W=self.W, coords=self.coords, plotting=self.plotting, gtype=self.gtype)


class DavidSensorNet(Graph):
    r"""
    Creates a sensor network

    Parameters
    ----------
    N: Number of vertices
        default is 64
    """

    def __init__(self, N=64):
        self.N = N

        if self.N == 64:
            david64 = PointsCloud("david64")
            self.W = david64.W
            self.N = david64.N
            self.coords = david64.coords

        elif self.N == 500:
            david500 = PointsCloud("david500")
            self.W = david500.W
            self.N = david500.N
            self.coords = david500.coords

        else:
            self.coords = np.random.rand(self.N, 2)

            target_dist_cutoff = -0.125*self.N/436.075+0.2183
            T = 0.6
            s = sqrt(-target_dist_cutoff**2/(2.*log(T)))
            d = gsp_distanz(np.conjugate(np.transpose(self.coords)))
            W = np.exp(-np.power(d, 2)/2.*s**2)
            W = np.where(W < T, 0, W)
            W -= np.diag(np.diag(W))
            self.W = sparse.lil_matrix(W)

        self.gtype = 'davidsensornet'
        self.plotting = {"limits": [0, 1, 0, 1]}

        super(DavidSensorNet, self).__init__(W=self.W, N=self.N, coords=self.coords, plotting=self.plotting, gtype=self.gtype)


class FullConnected(Graph):
    r"""
    Creates a fully connected graph

    Parameters
    ----------
    N: Number of vertices
        default 10
    """

    def __init__(self, N=10):
        self.N = N

        self.W = np.ones((self.N, self.N))-np.identity(self.N)

        tmp = np.arange(0, N).reshape(N, 1)
        self.coords = np.concatenate((np.cos(tmp*2*np.pi/self.N),
                                      np.sin(tmp*2*np.pi/self.N)),
                                     axis=1)
        self.plotting = {"limits": np.array([-1, 1, -1, 1])}
        self.gtype = "full"

        super(FullConnected, self).__init__(N=self.N, W=self.W, coords=self.coords, plotting=self.plotting, gtype=self.gtype)


class Logo(Graph):
    r"""
    Creates a graph with the GSP Logo
    """

    def __init__(self):
        logo = PointsCloud("logo")

        self.W = logo.W
        self.coords = logo.coords
        self.info = logo.info

        self.limits = np.array([0, 640, -400, 0])
        self.gtype = 'LogoGSP'

        self.plotting = {"vertex_color": np.array([200./255., 136./255., 204./255.]),
                         "edge_color": np.array([0, 136./255., 204./255.]),
                         "vertex_size": 20}

        super(Logo, self).__init__(W=self.W, coords=self.coords, gtype=self.gtype, limits=self.limits, plotting=self.plotting)


class Path(Graph):
    r"""
    Creates a path graph

    Parameters
    ----------
    N: Number of vertices
        default 32
    """

    def __init__(self, N=16):
        self.N = N

        inds_i = np.concatenate((np.arange(self.N-1), np.arange(1, self.N)),
                                axis=1)
        inds_j = np.concatenate((np.arange(1, self.N), np.arange(self.N-1)),
                                axis=1)

        self.W = sparse.csc_matrix((np.ones((2*(self.N - 1))),
                                    (inds_i, inds_j)),
                                   shape=(self.N, self.N))
        self.coords = np.concatenate((np.arange(1, self.N+1).reshape(self.N, 1),
                                     np.zeros((self.N, 1))),
                                    axis=1)
        self.plotting = {"limits": np.array([0, N+1, -1, 1])}
        self.gtype = "path"

        super(Path, self).__init__(W=self.W, coords=self.coords, plotting=self.plotting, gtype=self.gtype)


class RandomRing(Graph):
    r"""
    Creates a ring graph

    Parameters
    ----------
    N: Number of vertices
        default 64
    """

    def __init__(self, N=64):
        self.N = N

        position = np.sort(np.random.rand(self.N, 1), axis=0)

        weight = self.N*np.diff(self.N, axis=0)
        weightend = self.N*(1 + position[0] - position[-1])

        inds_j = np.conjugate(np.arange(2, self.N + 1).reshape(self.N-1, 1))
        inds_i = np.conjugate(np.arange(1, self.N).reshape(self.N-1, 1))

        W = sparse.csc_matrix((weight, (inds_i, inds_j)),
                              shape=(self.N, self.N))
        W[self.N, 1] = weightend

        self.W += np.conjugate(np.transpose(W))

        self.coords = np.concatenate((np.cos(position*2*np.pi),
                                      np.sin(position*2*np.pi)),
                                     axis=1)

        self.limits = np.array([-1, 1, -1, 1])
        self.gtype = 'random-ring'

        super(RandomRing, self).__init__(N=self.N, W=self.W, coords=self.coords, limits=self.limits, gtype=self.gtype)


class PointsCloud(object):
    r"""
    POINTCLOUD Load the parameters of models and the points
    Usage:  P = gsp_pointcloud(name)
            P = gsp_pointcloud(name, max_dim)

    Input parameters:
            name: the name of the point cloud to load ('airfoil', 'bunny', 'david64', 'david500', 'logo', 'two_moons')
            max_dim: the maximum dimensionality of the points (only valid for two_moons)
                default is 2

    Output parameters:
            The differents informations of the PointsCloud Loaded.

            bunny = PointsCloud('bunny')
            x = bunny.Xin

    'gsp_pointcloud( name, max_dim)' load pointcloud data and format it in
    a unified way as a set of points with each dimension in a different
    column

    Note that the bunny is the model from the Stanford Computer Graphics
    Laboratory see references.

    See also: gsp_nn_graph

    References: turk1994zippered
    """

    def __init__(self, pointcloudname, max_dim=2):
        if pointcloudname == "airfoil":
            airfoilmat = io.loadmat(os.path.dirname(os.path.realpath(__file__))
                                    + '/misc/airfoil.mat')
            self.i_inds = airfoilmat['i_inds']
            self.j_inds = airfoilmat['j_inds']
            self.x = airfoilmat['x']
            self.y = airfoilmat['y']

        elif pointcloudname == "bunny":
            bunnymat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) +
                                  '/misc/bunny.mat')
            self.Xin = bunnymat["bunny"]

        elif pointcloudname == "david64":
            david64mat = io.loadmat(os.path.dirname(os.path.realpath(__file__))
                                    + '/misc/david64.mat')
            self.W = david64mat["W"]
            self.N = david64mat["N"][0, 0]
            self.coords = david64mat["coords"]

        elif pointcloudname == "david500":
            david500mat = io.loadmat(os.path.dirname(os.path.realpath(__file__))
                                     + '/misc/david500.mat')
            self.W = david500mat["W"]
            self.N = david500mat["N"][0, 0]
            self.coords = david500mat["coords"]

        elif pointcloudname == "logo":
            logomat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) +
                                 '/misc/logogsp.mat')
            self.W = logomat["W"]
            self.coords = logomat["coords"]
            self.limits = np.array([0, 640, -400, 0])

            self.info = {"idx_g": logomat["idx_g"],
                         "idx_s": logomat["idx_s"],
                         "idx_p": logomat["idx_p"]}

        elif pointcloudname == "two_moons":
            twomoonsmat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/misc/two_moons.mat')
            if max_dim == -1:
                max_dim == 2
            self.Xin = twomoonsmat["features"][:max_dim].transpose()

        else:
            raise ValueError("This PointsCloud does not exist. Please verify you wrote the right name in lower case.")


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a: int
        Description.
    b: array_like
        Description.
    c: bool
        Description.

    Returns
    -------
    d: ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.graphs.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
