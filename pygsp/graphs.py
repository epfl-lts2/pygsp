# -*- coding: utf-8 -*-

r"""
This module implements principally graphs and some PointsClouds

* :class: `Graph` Main graph class

* :class: `PointsCloud` Class countaining all the PointsClouds
"""

import os
import os.path
import numpy as np
import random as rd
from math import ceil, sqrt, log, exp, floor, pi
from copy import deepcopy
from scipy import sparse, io, spatial

from pygsp import utils, plotting, operators


class PointsCloud(object):
    r"""
    Load the parameters of models and the points

    Parameters
    ----------
    name (string) : the name of the point cloud to load
        possible name: 'airfoil', 'bunny', 'david64', 'david500', 'logo', 'two_moons'
    max_dim (int) : the maximum dimensionality of the points (only valid for two_moons)
        default is 2

    Returns
    -------
    The differents informations of the PointsCloud Loaded.


    Examples
    --------
    >>> from pygsp import graphs
    >>> bunny = graphs.PointsCloud('bunny')
    >>> Xin = bunny.Xin

    Note
    ----
    The bunny is the model from the Stanford Computer Graphics Laboratory
    (see reference).


    Reference
    ----------
    :cite:`turk1994zippered`

    """

    def __init__(self, pointcloudname, max_dim=2):
        if pointcloudname == "airfoil":
            airfoilmat = io.loadmat(os.path.dirname(os.path.realpath(__file__))
                                    + '/misc/airfoil.mat')
            self.i_inds = airfoilmat['i_inds']
            self.j_inds = airfoilmat['j_inds']
            self.x = airfoilmat['x']
            self.y = airfoilmat['y']
            self.coords = np.concatenate((self.x, self.y), axis=1)

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

        elif pointcloudname == "minnesota":
            minnesotamat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/misc/minnesota.mat')
            self.A = minnesotamat["A"]
            self.labels = minnesotamat["labels"]
            self.coords = minnesotamat["xy"]

        elif pointcloudname == "two_moons":
            twomoonsmat = io.loadmat(os.path.dirname(os.path.realpath(__file__)) + '/misc/two_moons.mat')
            if max_dim == -1:
                max_dim == 2
            self.Xin = twomoonsmat["features"][:max_dim].transpose()

        else:
            raise ValueError("This PointsCloud does not exist. Please verify you wrote the right name in lower case.")


class Graph(object):
    r"""
    The main graph object

    It is used to initialize by default every missing field of the subgraphs
    It can also be used alone to initialize customs graphs

    Parameters
    ----------
    W (sparse) : weights matrix
        default is empty
    A (sparse) : adjancency matrix
        default is constructed with W
    N (int) : number of nodes
        default is the lenght of the first dimension of W
    d (float) : degree vector
        default
    Ne (int) : edge number
    gtype (strnig) : graph type
        default is "unknown"
    directed (bool) : whether the graph is directed
        default depending of the previous values
    lap_type (string) : laplacian type
        default is 'combinatorial'
    L (Ndarray): laplacian
    coords : Coordinates of the vertices
        default is np.array([0, 0])
    plotting (Dict): § dictionnary conataining the plotting parameters

    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> W = np.arange(4).reshape(2, 2)
    >>> G = graphs.Graph()

    """

    # All the parameters that needs calculation to be set
    # or not needed are set to None
    def __init__(self, W, A=None, N=None, d=None, Ne=None,
                 gtype='unknown', directed=None, coords=None,
                 lap_type='combinatorial', L=None, verbose=False,
                 plotting={}, **kwargs):

        self.gtype = gtype
        self.lap_type = lap_type
        self.verbose = verbose

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
            self.directed = utils.is_directed(self.W)
        if L is not None:
            self.L = L
        else:
            self.L = operators.create_laplacian(self)

        # Plotting default parameters
        self.plotting = {}
        if 'edge_width' in plotting:
            self.plotting['edge_width'] = plotting['edge_width']
        else:
            self.plotting['edge_width'] = 1
        if 'edge_color' in plotting:
            self.plotting['edge_color'] = plotting['edge_color']
        else:
            self.plotting['edge_color'] = np.array([255, 88, 41])/255.
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

    def deep_copy_graph(self):
        r"""
        TODO write doc
        """
        return deepcopy(self)

    def copy_graph_attributes(self, ctype=True, Gn=None):
        r"""
        copy graph attributes copy the parameter of the graph

        Parameters
        ----------:
        G : Graph structure
        ctype (bool): flag to select what to copy
            Default is True
        Gn : Graph structure (optional)

        Returns
        -------
        Gn : Partial graph structure

        Examples
        --------
        >>> from pygsp import graphs
        >>> Torus = graphs.Torus()
        >>> G = graphs.TwoMoons()
        >>> G.copy_graph_attributes(type=0, Gn=Torus);
        """
        # if no Gn given
        if not Gn:
            if ctype:
                Gn = Graph(lap_type=G.lap_type, plotting=G.plotting, limits=G.limits)
            else:
                Gn = Graph(lap_type=G.lap_type, plotting=G.plotting)

            return Gn

        # if Gn given.
        if hasattr(G, 'lap_type'):
            Gn.lap_type = G.lap_type

        if hasattr(G, 'plotting'):
            Gn.plotting = G.plotting

        if ctype:
            if hasattr(G, 'coords'):
                Gn.coords = G.coords
        else:
            if hasattr(Gn.plotting, 'limits'):
                del GN.plotting['limits']

    def separate_graph(self):
        r"""
        TODO write func & doc
        """
        raise NotImplementedError("Not implemented yet")

    def subgraph(self, c):
        r"""
        Create a subgraph from G

        Parameters
        ----------
        G (graph) : Original graph
        c (int) : Node to keep

        Returns
        -------
        subG (graph) : Subgraph

        Examples
        --------
        >>> from pygsp import graphs
        >>> import numpy as np
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> c = 10
        >>> subG = graphs.Graph.subgraph(G, c)

        This function create a subgraph from G taking only the node in c.

        """

        sub_G = self
        sub_G.W = self.W[c, c]
        try:
            sub_G.N = len(c)
        except TypeError:
            sub_G.N = 1

        sub_G.gtype = "sub-" + self.gtype

        return sub_G


class NNGraph(Graph):
    r"""
    Creates a graph from a pointcloud

    Parameters
    ----------
    Xin (nunpy Array) : Input Points
    use_flann : Whether flann method should be used (knn is otherwise used)
        default is False
        (not implemented yet)
    center (bool) : center the data
        default is True
    rescale (bool) : rescale the data (in a 1-ball)
        default is True
    k (int) : number of neighbors for knn
        default is 10
    sigma (float) : variance of the distance kernel
        default is 0.1
    epsilon (float) : radius for the range search
        default is 0.01
    gtype (string) : the type of graph
            default is "knn"

    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> Xin = np.arange(9)
    >>> Xin = Xin.reshape(3,3)
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

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if self.center:
            Xout = self.Xin - np.kron(np.ones((N, 1)),
                                      np.mean(self.Xin, axis=0))

        if self.rescale:
            bounding_radius = 0.5*np.linalg.norm(np.amax(Xout, axis=0) -
                                                 np.amin(Xout, axis=0), 2)
            scale = np.power(N, 1./float(min(d, 3)))/10.
            Xout *= scale/bounding_radius

        if self.NNtype == "knn":
            spi = np.zeros((N*k))
            spj = np.zeros((N*k))
            spv = np.zeros((N*k))

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
                spv[i*k:(i+1)*k] = np.exp(-np.power(D[i, 1:], 2) /
                                          float(self.sigma))

            W = sparse.csc_matrix((spv, (spi, spj)),
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
                spv[start:start+leng] = np.exp(-np.power(D[i][1:], 2) /
                                               float(self.sigma))
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
        if utils.is_directed(W):
            W = utils.symetrize(W, symetrize_type=self.symetrize_type)
        else:
            pass

        self.N = N
        self.W = W
        self.coords = Xout

        super(NNGraph, self).__init__(N=self.N, gtype=self.gtype, W=self.W,
                                      plotting=self.plotting,
                                      coords=self.coords, **kwargs)


class Bunny(NNGraph):
    r"""
    Create a graph of the stanford bunny

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Bunny()

    References
    ----------
    :cite:`turk1994zippered`

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
    radius (float) : edge lenght
        default is 1
    nb_pts (int) : number of vertices
        default is 300
    nb_dim (int) : dimension
        default is 3
    sampling (string) : variance of the distance kernel
        default is 'random'
        (Can now only be 'random')

    Examples
    --------
    >>> from pygsp import graphs
    >>> radius = 5
    >>> G = graphs.Cube(radius=radius)

    """

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling="random",
                 **kwargs):
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
                n = floor(self.nb_pts/6.)

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

        self.NNtype = 'knn'
        self.gtype = "Cube"
        self.k = 10

        super(Cube, self).__init__(Xin=pts, k=self.k, NNtype=self.NNtype,
                                   gtype=self.gtype, **kwargs)


class Sphere(NNGraph):
    r"""
    Creates a spherical-shaped graph

    Parameters
    ----------
    radius (flaot) : radius of the sphere
        default is 1
    nb_pts (int) : number of vertices
        default is 300
    nb_dim (int) : dimension
        default is 3
    sampling (sting) : variance of the distance kernel
        default is 'random'
        (Can now only be 'random')

    Examples
    --------
    >>> from pygsp import graphs
    >>> radius = 5
    >>> G = graphs.Sphere(radius=radius)

    """

    def __init__(self, radius=1, nb_pts=300, nb_dim=3, sampling="random",
                 **kwargs):
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling

        if self.sampling == "random":
            pts = np.random.normal(0, 1, (self.nb_pts, self.nb_dim))
            for i in range(self.nb_pts):
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
    moontype (string): You have the freedom to chose if you want to create a standard two_moons graph or a synthetised one (default is 'standard').
        * 'standard' : create a two_moons graph from a based graph.
            sigmag (flaot) : variance of the distance kernel
                default is 0.05

        * 'synthetised' : create a synthetised two_moon
            sigmag (flaot) : variance of the distance kernel
                default is 0.05
            N (int) : Number of vertices
                default is 2000
            sigmad (flaot) : variance of the data (do not set it to high or you won't see anything)
                default is 0.05
            d (flaot) : distance of the two moons
                default is 0.5

    Examples
    --------
    >>> from pygsp import graphs
    >>> G1 = graphs.TwoMoons(moontype='standard')
    >>> G2 =  graphs.TwoMoons(moontype='synthetised', N=1000, sigmad=0.1, d=1)

    """

    def __init__(self, moontype="standard", sigmag=0.05, N=400, sigmad=0.07,
                 d=0.5):

        def create_arc_moon(N, sigmad, d, number):
            phi = np.random.rand(N, 1)*np.pi
            r = 1
            rb = sigmad*np.random.normal(size=(N, 1))
            ab = np.random.rand(N, 1)*2*np.pi
            b = rb*np.exp(1j*ab)
            bx = np.real(b)
            by = np.imag(b)

            if number == 1:
                moonx = np.cos(phi)*r + bx + 0.5
                moony = -np.sin(phi)*r + by - (d-1)/2.
            elif number == 2:
                moonx = np.cos(phi)*r + bx - 0.5
                moony = np.sin(phi)*r + by + (d-1)/2.

            return np.concatenate((moonx, moony), axis=1)

        self.k = 5
        self.sigma = sigmag

        if moontype == "standard":
            two_moons = PointsCloud("two_moons")
            self.Xin = two_moons.Xin

            self.gtype = "Two Moons standard"
            self.labels = 2*(np.where(np.arange(1, N+1).reshape(N, 1) > 1000,
                                      1, 0) + 1)

            super(TwoMoons, self).__init__(Xin=self.Xin, sigma=sigmag,
                                           labels=self.labels, k=self.k,
                                           gtype=self.gtype)

        elif moontype == 'synthetised':
            self.gtype = "Two Moons synthetised"

            N1 = floor(N/2.)
            N2 = N - N1

            # Moon 1
            Coordmoon1 = create_arc_moon(N1, sigmad, d, 1)

            # Moon 2
            Coordmoon2 = create_arc_moon(N2, sigmad, d, 2)

            self.Xin = np.concatenate((Coordmoon1, Coordmoon2))
            self.labels = 2*(np.where(np.arange(1, N+1).reshape(N, 1) >
                                      N1, 1, 0) + 1)

            super(TwoMoons, self).__init__(Xin=self.Xin, sigma=sigmag,
                                           labels=self.labels, k=self.k,
                                           gtype=self.gtype)


# Need M
class Grid2d(Graph):
    r"""
    Creates a 2 dimensional grid graph

    Parameters
    ----------
    Nv (int) : Number of vertices along the first dimension
        default is 16
    Mv (int) : Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Grid2d(Nv=32)

    """

    def __init__(self, Nv=16, Mv=None, **kwargs):
        if not Mv:
            Mv = Nv

        # Create weighted adjacency matrix
        K = 2*(Nv-1)
        J = 2*(Mv-1)

        i_inds = np.zeros((K*Mv + J*Nv), dtype=float)
        j_inds = np.zeros((K*Mv + J*Nv), dtype=float)

        for i in range(Mv):
            i_inds[i*K + np.arange(K)] = i*Nv + np.concatenate((np.arange(Nv-1), np.arange(1, Nv)))
            j_inds[i*K + np.arange(K)] = i*Nv + np.concatenate((np.arange(1, Nv), np.arange(Nv-1)))

        for i in range(Mv-1):
            i_inds[(K*Mv) + i*2*Nv + np.arange(2*Nv)] = np.concatenate((i*Nv + np.arange(Nv), (i+1)*Nv + np.arange(Nv)))
            j_inds[(K*Mv) + i*2*Nv + np.arange(2*Nv)] = np.concatenate(((i+1)*Nv + np.arange(Nv), i*Nv + np.arange(Nv)))

        self.W = sparse.csc_matrix((np.ones((K*Mv+J*Nv)), (i_inds, j_inds)), shape=(Mv*Nv, Mv*Nv))

        xtmp = np.kron(np.ones((Mv, 1)), (np.arange(Nv)/float(Nv)).reshape(Nv,
                                                                           1))
        ytmp = np.sort(np.kron(np.ones((Nv, 1)),
                               np.arange(Mv)/float(Mv)).reshape(Mv*Nv, 1),
                       axis=0)

        self.coords = np.concatenate((xtmp, ytmp), axis=1)

        self.N = Nv * Mv
        self.Nv = Nv
        self.Mv = Mv
        self.gtype = '2d-grid'
        self.plotting = {"limits": np.array([-1./self.Nv, 1 + 1./self.Nv,
                                             1./self.Mv, 1 + 1./self.Mv]),
                         "vertex_size": 30}

        super(Grid2d, self).__init__(N=self.N, W=self.W, gtype=self.gtype,
                                     plotting=self.plotting,
                                     coords=self.coords, **kwargs)


class Torus(Graph):
    r"""
    Creates a Torus graph

    Parameters
    ----------
    Nv (int) : Number of vertices along the first dimension
        default is 16
    Mv (int) : Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> Nv = 32
    >>> G = graphs.Torus(Nv=Nv)

    """

    def __init__(self, Nv=16, Mv=None, **kwargs):

        if not Mv:
            Mv = Nv

        # Create weighted adjancency matrix
        K = 2 * Nv
        J = 2 * Mv
        i_inds = np.zeros((K*Mv + J*Nv), dtype=float)
        j_inds = np.zeros((K*Mv + J*Nv), dtype=float)

        for i in range(Mv):
            i_inds[i*K + np.arange(K)] = i*Nv + np.concatenate((np.array([Nv-1]), np.arange(Nv-1), np.arange(Nv)))
            j_inds[i*K + np.arange(K)] = i*Nv + np.concatenate((np.arange(Nv), np.array([Nv-1]), np.arange(Nv-1)))

        for i in range(Mv-1):
            i_inds[K*Mv + i*2*Nv + np.arange(2*Nv)] = np.concatenate((i*Nv + np.arange(Nv), (i+1)*Nv + np.arange(Nv)))
            j_inds[K*Mv + i*2*Nv + np.arange(2*Nv)] = np.concatenate(((i+1)*Nv + np.arange(Nv), i*Nv + np.arange(Nv)))

        i_inds[K*Mv + (Mv-1)*2*Nv + np.arange(2*Nv)] = np.concatenate((np.arange(Nv), (Mv-1)*Nv + np.arange(Nv)))
        j_inds[K*Mv + (Mv-1)*2*Nv + np.arange(2*Nv)] = np.concatenate(((Mv-1)*Nv + np.arange(Nv), np.arange(Nv)))

        self.W = sparse.csc_matrix((np.ones((K*Mv+J*Nv)), (i_inds, j_inds)),
                                   shape=(Mv*Nv, Mv*Nv))

        # Create Coordinate
        T = 1.5 + np.sin(np.arange(Mv)*2*np.pi/Mv).reshape(1, Mv)
        U = np.cos(np.arange(Mv)*2*np.pi/Mv).reshape(1, Mv)
        xtmp = np.cos(np.arange(Nv).reshape(Nv, 1)*2*np.pi/Nv)*T
        ytmp = np.sin(np.arange(Nv).reshape(Nv, 1)*2*np.pi/Nv)*T
        ztmp = np.kron(np.ones((Nv, 1)), U)
        self.coords = np.concatenate((np.reshape(xtmp, (Mv*Nv, 1), order='F'),
                                      np.reshape(ytmp, (Mv*Nv, 1), order='F'),
                                      np.reshape(ztmp, (Mv*Nv, 1), order='F')),
                                     axis=1)
        self.Nv = Nv
        self.Mv = Nv
        self.directed = False
        self.gtype = 'Torus'
        self.plotting = {"vertex_size": 30,
                         "limits": np.array([-2.5, 2.5, -2.5, 2.5, -2.5, 2.5])}

        super(Torus, self).__init__(W=self.W, directed=self.directed,
                                    gtype=self.gtype, coords=self.coords,
                                    plotting=self.plotting, **kwargs)


# Need K
class Comet(Graph):
    r"""
    Creates a Comet graph

    Parameters
    ----------
    Nv (int) : Number of vertices along the first dimension
        default is 16
    Mv (int) : Number of vertices along the second dimension
        default is Nv

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Comet() # (== graphs.Comet(Nv=32, k=12))

    """

    def __init__(self, Nv=32, k=12, **kwargs):

        # Create weighted adjancency matrix
        i_inds = np.concatenate((np.zeros((k)), np.arange(k)+1,
                                 np.arange(k, Nv-1),
                                 np.arange(k+1, Nv)))
        j_inds = np.concatenate((np.arange(k)+1, np.zeros((k)),
                                 np.arange(k+1, Nv),
                                 np.arange(k, Nv-1)))

        self.W = sparse.csc_matrix((np.ones((np.size(i_inds))),
                                    (i_inds, j_inds)),
                                   shape=(Nv, Nv))

        tmpcoords = np.zeros((Nv, 2))
        inds = np.arange(k)+1
        tmpcoords[1:k+1, 0] = np.cos(inds*2*np.pi/k)
        tmpcoords[1:k+1, 1] = np.sin(inds*2*np.pi/k)
        tmpcoords[k+1:, 0] = np.arange(1, Nv-k)+1

        self.coords = tmpcoords
        self.Nv = Nv
        self.k = k
        self.gtype = 'Comet'
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
    k (int) : 2^k points on each side of the grid of vertices
        default 6

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.LowStretchTree(k=3)

    # >>> plotting.plot_graph(G)
    """

    def __init__(self, k=6, **kwargs):

        XCoords = np.array([1, 2, 1, 2])
        YCoords = np.array([1, 1, 2, 2])

        ii = np.array([0, 0, 1, 2, 2, 3])
        jj = np.array([1, 2, 1, 3, 0, 2])

        for p in range(1, k):
            ii = np.concatenate((ii, ii + 4**p, ii + 2*4**p,
                                 ii + 3*4**p, [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1],
                                 [5/3.*4**p + 1/3. - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1], [3*4**p]))
            jj = np.concatenate((jj, jj + 4**p, jj + 2*4**p, jj + 3*4**p,
                                 [5./3*4**p + 1/3. - 1],
                                 [4**p + (4**(p+1) + 2)/3. - 1],
                                 [3*4**p], [4**p - 1], [4**p - 1],
                                 [4**p + (4**(p+1)+2)/3. - 1]))

            YCoords = np.kron(np.ones((2)), YCoords)
            YCoords = np.concatenate((YCoords, YCoords + 2**p))

            XCoords = np.concatenate((XCoords, XCoords + 2**p))
            XCoords = np.kron(np.ones((2)), XCoords)

        self.coords = np.concatenate((np.expand_dims(XCoords, axis=1),
                                      np.expand_dims(YCoords, axis=1)),
                                     axis=1)

        self.limits = np.array([0, 2**k+1, 0, 2**k+1])
        self.N = (2**k)**2
        self.W = sparse.csc_matrix((np.ones((np.shape(ii))), (ii, jj)))
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
    The random regular graph has the property that every nodes is connected to
    'k' other nodes.

    Parameters
    ----------
    N (int) : Number of nodes
        default is 64
    k (int) : Number of connections of each nodes
        default is 6

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.RandomRegular()

    """

    def __init__(self, N=64, k=6, verbose=False, **kwargs):

        def isRegularGraph(A):
            r"""
            This fonction prints a message describing the problem of a given
            sparse matrix

            Inputs
            ------
            A (Sparse matrix)

            """

            msg = "The given matrix "

            # check if the sparse matrix is in a good format
            if A.getformat() == 'lil' or A.getformat() == 'dia' or A.getformat() == 'bok':
                A = A.tocsc()

            # check symmetry
            tmp = (A-A.getH())
            if np.sum((tmp.getH()*tmp).diagonal()) > 0:
                msg += "is not symetric, "

            # check parallel edged
            if A.max(axis=None) > 1:
                msg += "has parallel edges, "

            # check that d is d-regular
            d_vec = A.sum(axis=0)
            if np.min(d_vec) < d_vec[:, 0] and np.max(d_vec) > d_vec[:, 0]:
                msg += "not d-regular, "

            # check that g doesn't contain any loops
            if A.diagonal().any() > 0:
                msg += "has self loops, "

            else:
                msg += "is ok"

            if verbose:
                print(msg)

        def createRandRegGraph(vertNum, deg):
            r"""
            creates a simple d-regular undirected graph
            simple = without loops or double edges
            d-reglar = each vertex is adjecent to d edges

            Parameters
            ----------
            vertNum : number of vertices
            deg : the degree of each vertex

            Returns
            -------
            A (sparse) : representation of the graph

            Algorithm
            ---------
            "The pairing model": create n*d 'half edges'.
            repeat as long as possible: pick a pair of half edges
            and if it's legal (doesn't creat a loop nor a double edge)
            add it to the graph

            Reference
            ---------
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.7957&rep=rep1&type=pdf
            (This code has been adapted from matlab to python)
            """

            n = vertNum
            d = deg
            matIter = 10

            # continue until a proper graph is formed
            if (n*d) % 2 == 1:
                raise ValueError("createRandRegGraph input err:\
                                 n*d must be even!")

            # a list of open half-edges
            U = np.kron(np.ones((d)), np.arange(n))

            # the graphs adajency matrix
            A = sparse.lil_matrix(np.zeros((n, n)))

            edgesTested = 0
            repetition = 1

            # check that there are no loops nor parallel edges
            while np.size(U) != 0 and repetition < matIter:
                edgesTested += 1

                # print progess
                if edgesTested % 5000 == 0:
                    print("createRandRegGraph() progress: edges=%d/%d\n" %
                          (edgesTested, n*d))

                # chose at random 2 half edges
                i1 = floor(rd.random()*np.shape(U)[0])
                i2 = floor(rd.random()*np.shape(U)[0])
                v1 = U[i1]
                v2 = U[i2]

                # check that there are no loops nor parallel edges
                if v1 == v2 or A[v1, v2] == 1:
                    # restart process if needed
                    if edgesTested == n*d:
                        repetition = repetition + 1
                        edgesTested = 0
                        U = np.kron(np.ones((d)), np.arange(n))
                        A = sparse.lil_matrix(np.zeros((n, n)))
                else:
                    # add edge to graph
                    A[v1, v2] = 1
                    A[v2, v1] = 1

                    # remove used half-edges
                    v = sorted([i1, i2])
                    U = np.concatenate((U[1:v[0]], U[v[0]+1:v[1]], U[v[1]+1:]))

            isRegularGraph(A)

            return A

        self.N = N
        self.k = k

        self.gtype = "random_regular"
        self.W = createRandRegGraph(self.N, self.k)

        super(RandomRegular, self).__init__(W=self.W, gtype=self.gtype,
                                            **kwargs)


class Ring(Graph):
    r"""
    Creates a ring graph

    Parameters
    ----------
    N (int) : Number of vertices
        default is 64
    k (int) : Number of neighbors in each directions
        default is 1

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Ring()

    """

    def __init__(self, N=64, k=1, **kwargs):

        if k > N/2.:
            raise ValueError("Too many neighbors requested.")

        # Create weighted adjancency matrix
        if k == N/2.:
            num_edges = N*(k-1) + N/2.
        else:
            num_edges = N*k

        i_inds = np.zeros((2*num_edges))
        j_inds = np.zeros((2*num_edges))

        all_inds = np.arange(N)
        for i in range(min(k, floor((N-1)/2.))):
            i_inds[i*2*N + np.arange(N)] = all_inds
            j_inds[i*2*N + np.arange(N)] = np.remainder(all_inds + i + 1, N)
            i_inds[(i*2+1)*N + np.arange(N)] = np.remainder(all_inds + i + 1, N)
            j_inds[(i*2+1)*N + np.arange(N)] = all_inds

        if k == N/2.:
            i_inds[2*N*(k-1) + np.arange(N)] = all_inds
            i_inds[2*N*(k-1) + np.arange(N)] = np.remainder(all_inds + k + 1, N)

        self.W = sparse.csc_matrix((np.ones((2*num_edges)), (i_inds, j_inds)),
                                   shape=(N, N))

        self.coords = np.concatenate((np.cos(np.arange(N).reshape(N, 1)*2*np.pi/float(N)),
                                      np.sin(np.arange(N).reshape(N, 1)*2*np.pi/float(N))),
                                     axis=1)

        self.plotting = {"limits": np.array([-1, 1, -1, 1])}

        if k == 1:
            self.gtype = "ring"
        else:
            self.gtype = "k-ring"

        self.N = N
        self.k = k

        super(Ring, self).__init__(W=self.W, N=self.N, gtype=self.gtype,
                                   coords=self.coords, plotting=self.plotting,
                                   **kwargs)


# Need params
class Community(Graph):
    r"""
    Create a community graph

    Parameters
    ----------
    N (int) : Number of nodes
        default is 256
    Nc (int) : Number of communities
        default is round(sqrt(N)/2)
    com_sizes (int) : Size of the communities
        default is is random
    min_comm (int) : Minimum size of the communities
        default is round(N/Nc/3)
    min_deg (int) : Minimum degree of each node
        default is round(min_comm/2) (not implemented yet)
    verbose (int) : Verbosity output
        default is 1
    size_ratio (float) : Ratio between the radius of world and the radius of communities
        default is 1
    world_density (float) : Probability of a random edge between any pair of edges
        default is 1/N

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Community()

    """

    def __init__(self, N=256, Nc=None, com_sizes=np.array([]), min_com=None,
                 min_deg=None, verbose=1, size_ratio=1, world_density=None):

        # Initialisation of the parameters
        if not Nc:
            Nc = int(round(sqrt(N)/2.))

        if len(com_sizes) != 0:
            if np.sum(com_sizes) != N:
                raise ValueError("GSP_COMMUNITY: The sum of the community \
                                 sizes has to be equal to N")

        if not min_com:
            min_com = round(float(N) / Nc / 3.)

        if not min_deg:
            min_deg = round(min_com/2.)

        if not world_density:
            world_density = 1./N

        # Begining
        if np.shape(com_sizes)[0] == 0:
            x = N - (min_com - 1)*Nc - 1
            com_lims = np.sort(np.resize(np.random.permutation(int(x)), (Nc-1.))) + 1
            com_lims += np.cumsum((min_com-1)*np.ones(np.shape(com_lims)))
            com_lims = np.concatenate((np.array([0]), com_lims, np.array([N])))
            com_sizes = np.diff(com_lims)

        if verbose > 2:
                X = np.zeros((10000, Nc + 1))
                # pick randomly param.Nc-1 points to cut the rows in communtities:
                for i in range(10000):
                    com_lims_tmp = np.sort(np.resize(np.random.permutation(int(x)), (Nc-1.))) + 1
                    com_lims_tmp += np.cumsum((min_com-1)*np.ones(np.shape(com_lims_temp)))
                    X[i, :] = np.concatenate((np.array([0]), com_lims_tmp, np.array([N])))
                dX = np.transpose(np.diff(np.transpose(X)))
                for i in range(int(Nc)):
                    # TODO figure; hist(dX(:,i), 100); title('histogram of row community size'); end
                    pass
                del X
                del com_lims_tmp

        rad_world = size_ratio*sqrt(N)
        com_coords = rad_world*np.concatenate((-np.expand_dims(np.cos(2*np.pi*(np.arange(Nc) + 1)/Nc), axis=1),
                                               np.expand_dims(np.sin(2*np.pi*(np.arange(Nc) + 1)/Nc), axis=1)),
                                              axis=1)

        coords = np.ones((N, 2))

        # create uniformly random points in the unit disc
        for i in range(N):
            # use rejection sampling to sample from a unit disc (probability = pi/4)
            while np.linalg.norm(coords[i], 2) >= 0.5:
                # sample from the square and reject anything outside the circle
                coords[i] = rd.random()-0.5, rd.random()-0.5

        info = {"node_com": np.zeros((N, 1))}

        # add the offset for each node depending on which community it belongs to
        for i in range(int(Nc)):
            com_size = com_sizes[i]
            rad_com = sqrt(com_size)

            node_ind = np.arange(com_lims[i], com_lims[i+1])
            coords[node_ind] = rad_com*coords[node_ind] + com_coords[i]
            info["node_com"] = i

        D = utils.distanz(np.transpose(coords))
        W = np.exp(-np.power(D, 2))
        W = np.where(W < 1e-3, 0, W)

        # When we make W symetric, the density get bigger (because we add a ramdom number of values)
        world_density = world_density/float(2-1./N)

        W = W + np.abs(sparse.rand(N, N, density=world_density))
        # W need to be symetric.
        w = (W + W.getH())/2.
        W = np.where(np.abs(W) > 0, 1, W).astype(float)

        self.W = sparse.coo_matrix(W)
        self.gtype = "Community"
        self.coords = coords
        self.N = N
        self.Nc = Nc

        # return additional info about the communities
        info["com_lims"] = com_lims
        info["com_coords"] = com_coords
        info["com_sizes"] = com_sizes
        self.info = info

        super(Community, self).__init__(W=self.W, gtype=self.gtype,
                                        coords=self.coords, info=self.info)


class Minnesota(Graph):
    r"""
    Create a community graph

    Parameters
    ----------
    connect (bool) : change the graph to be connected.
        default is True (--> default minnesota graph is coneected)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Minnesota()

    """

    def __init__(self, connect=True):
        minnesota = PointsCloud('minnesota')

        self.N = np.shape(minnesota.A)[0]
        self.coords = minnesota.coords
        self.plotting = {"limits": np.array([-98, -89, 43, 50]),
                         "vertex_size": 30}

        if connect:
            # Edit adjacency matrix
            A = minnesota.A.tolil()

            # clean minnesota graph
            A.setdiag(0)

            # missing edge needed to connect graph
            A[349, 355] = 1
            A[355, 349] = 1

            # change a handful of 2 values back to 1
            A[86, 88] = 1
            A[86, 88] = 1
            A[345, 346] = 1
            A[346, 345] = 1
            A[1707, 1709] = 1
            A[1709, 1707] = 1
            A[2289, 2290] = 1
            A[2290, 2289] = 1

            self.W = A
            self.gtype = 'minnesota'

        else:
            self.W = A
            self.gtype = 'minnesota-disconnected'

        super(Minnesota, self).__init__(W=self.W, gtype=self.gtype,
                                        coords=self.coords, N=self.N,
                                        plotting=self.plotting)


class Sensor(Graph):
    r"""
    Creates a random sensor graph

    Parameters
    ----------
    N (int) : Number of nodes
        default is 64
    Nc (int) : Minimum number of connections
        default is 1
    regular (bool) : Flag to fix the number of connections to nc
        default is False
    verbose (bool) : Verbosity parameter
        default is True
    n_try (int) : Number of attempt to create the graph
        default is 50
    distribute (bool) : To distribute the points more evenly
        default is False
    connected (bool): To force the graph to be connected
        default is True

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Sensor(N=300)

    """

    def __init__(self, N=64, Nc=2, regular=False, verbose=1, n_try=50,
                 distribute=False, connected=True, **kwargs):

        self.N = N
        self.Nc = Nc
        self.regular = regular
        self.verbose = verbose
        self.n_try = n_try
        self.distribute = distribute
        self.connected = connected

        def create_weight_matrix(N, param_distribute, param_regular, param_Nc):
            XCoords = np.zeros((N, 1))
            YCoords = np.zeros((N, 1))

            if param_distribute:
                mdim = int(ceil(sqrt(N)))
                for i in range(mdim):
                    for j in range(mdim):
                        if i*mdim + j < N:
                            XCoords[i*mdim + j] = np.array(1./float(mdim)*np.random.rand() + i/float(mdim))
                            YCoords[i*mdim + j] = np.array(1./float(mdim)*np.random.rand() + j/float(mdim))

            # take random coordinates in a 1 by 1 square
            else:
                XCoords = np.random.rand(N, 1)
                YCoords = np.random.rand(N, 1)

            Coords = np.concatenate((XCoords, YCoords), axis=1)

            # Compute the distanz between all the points
            target_dist_cutoff = 2*N**(-0.5)
            T = 0.6
            s = sqrt(-target_dist_cutoff**2/(2*log(T)))
            d = utils.distanz(x=np.transpose(Coords))
            W = np.exp(-d**2/(2.*s**2))
            W -= np.diag(np.diag(W))

            if param_regular:
                W = get_nc_connection(W, param_Nc)

            else:
                W2 = get_nc_connection(W, param_Nc)
                W = np.where(W < T, 0, W)
                W = np.where(W2 > 0, W2, W)

            return W, Coords

        def get_nc_connection(W, param_nc):
            Wtmp = W
            W = np.zeros(np.shape(W))

            for i in range(np.shape(W)[0]):
                l = Wtmp[i]
                for j in range(param_nc):
                    val = np.max(l)
                    ind = np.argmax(l)
                    W[i, ind] = val
                    l[ind] = 0

            W = (W + np.transpose(np.conjugate(W)))/2.

            return W

        if self.connected:
            for x in range(self.n_try):
                W, Coords = create_weight_matrix(self.N,
                                                 self.distribute,
                                                 self.regular,
                                                 self.Nc)

                self.W = W

                if utils.check_connectivity(self):
                    break

                elif x == self.n_try-1:
                    print("Warning! Graph is not connected")

        else:
            W, Coords = create_weight_matrix(self.N, self.distribute,
                                             self.regular, self.Nc)

        W = sparse.lil_matrix(W)
        self.W = (W + W.getH())/2.
        self.coords = Coords

        if self.regular:
            self.gtype = "regular sensor"
        else:
            self.gtype = "sensor"
        self.directed = False

        self.plotting = {"limits": np.array([0, 1, 0, 1])}

        super(Sensor, self).__init__(W=self.W, N=self.N, coords=self.coords,
                                     plotting=self.plotting, gtype=self.gtype,
                                     directed=self.directed, **kwargs)


# Need nothing
class Airfoil(Graph):
    r"""
    Creates the aifoil graph

    Parameters
    ----------
    None

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Airfoil()

    """

    def __init__(self):

        airfoil = PointsCloud("airfoil")
        i_inds = airfoil.i_inds
        j_inds = airfoil.j_inds

        A = sparse.coo_matrix((np.ones((12289)),
                              (np.reshape(i_inds-1, (12289)),
                               np.reshape(j_inds-1, (12289)))),
                              shape=(4253, 4253))
        self.W = (A + A.getH())/2.

        x = airfoil.x
        y = airfoil.y

        self.coords = airfoil.coords
        self.gtype = 'Airfoil'
        self.plotting = {"limits": np.array([-1e-4, 1.01*np.max(x), -1e-4, 1.01*np.max(y)]),
                         "vertex_size": 30}

        super(Airfoil, self).__init__(W=self.W, coords=self.coords,
                                      plotting=self.plotting, gtype=self.gtype)


class DavidSensorNet(Graph):
    r"""
    Creates a sensor network

    Parameters
    ----------
    N (int): Number of vertices
        default is 64

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.DavidSensorNet(N=500)

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
            d = utils.distanz(np.conjugate(np.transpose(self.coords)))
            W = np.exp(-np.power(d, 2)/2.*s**2)
            W = np.where(W < T, 0, W)
            W -= np.diag(np.diag(W))
            self.W = sparse.lil_matrix(W)

        self.gtype = 'davidsensornet'
        self.plotting = {"limits": [0, 1, 0, 1]}

        super(DavidSensorNet, self).__init__(W=self.W, plotting=self.plotting,
                                             N=self.N, coords=self.coords,
                                             gtype=self.gtype)


class FullConnected(Graph):
    r"""
    Creates a fully connected graph

    Parameters
    ----------
    N (int) : Number of vertices
        default 10

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.FullConnected(N=5)

    """

    def __init__(self, N=10):

        tmp = np.arange(0, N).reshape(N, 1)

        self.coords = np.concatenate((np.cos(tmp*2*np.pi/N),
                                      np.sin(tmp*2*np.pi/N)),
                                     axis=1)
        self.W = np.ones((N, N))-np.identity(N)
        self.N = N
        self.gtype = "full"
        self.plotting = {"limits": np.array([-1, 1, -1, 1])}

        super(FullConnected, self).__init__(W=self.W, plotting=self.plotting,
                                            N=self.N, coords=self.coords,
                                            gtype=self.gtype)


class Logo(Graph):
    r"""
    Creates a graph with the GSP Logo

    Parameters
    ----------
    None

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Logo()

    """

    def __init__(self):
        logo = PointsCloud("logo")

        self.W = logo.W
        self.coords = logo.coords
        self.info = logo.info

        self.limits = np.array([0, 640, -400, 0])
        self.gtype = 'LogoGSP'

        self.plotting = {"vertex_color": np.array([200./255, 136./255, 204./255]),
                         "edge_color": np.array([0, 136./255, 204./255]),
                         "vertex_size": 20}

        super(Logo, self).__init__(plotting=self.plotting, coords=self.coords,
                                   gtype=self.gtype, limits=self.limits,
                                   W=self.W)


class Path(Graph):
    r"""
    Creates a path graph

    Parameters
    ----------
    N (int) : Number of vertices
        default 32

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Path(N=16)

    """

    def __init__(self, N=16):

        inds_i = np.concatenate((np.arange(N-1), np.arange(1, N)))
        inds_j = np.concatenate((np.arange(1, N), np.arange(N-1)))

        self.W = sparse.csc_matrix((np.ones((2*(N - 1))), (inds_i, inds_j)),
                                   shape=(N, N))
        self.coords = np.concatenate((np.expand_dims(np.arange(N)+1, axis=1),
                                      np.zeros((N, 1))),
                                     axis=1)
        self.plotting = {"limits": np.array([0, N+1, -1, 1])}
        self.gtype = "path"
        self.N = N

        super(Path, self).__init__(W=self.W, coords=self.coords,
                                   plotting=self.plotting, gtype=self.gtype)


class RandomRing(Graph):
    r"""
    Creates a ring graph

    Parameters
    ----------
    N (int) : Number of vertices
        default 64

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.RandomRing(N=16)

    """

    def __init__(self, N=64):

        position = np.sort(np.random.rand(N), axis=0)

        weight = N*np.diff(position)
        weightend = N*(1 + position[0] - position[-1])

        inds_j = np.arange(1, N)
        inds_i = np.arange(N-1)

        W = sparse.csc_matrix((weight, (inds_i, inds_j)), shape=(N, N))
        W = W.tolil()
        W[N-1, 0] = weightend

        self.W = W + W.getH()

        self.coords = np.concatenate((np.expand_dims(np.cos(position*2*np.pi),
                                      axis=1),
                                      np.expand_dims(np.sin(position*2*np.pi),
                                      axis=1)),
                                     axis=1)

        self.N = N
        self.limits = np.array([-1, 1, -1, 1])
        self.gtype = 'random-ring'

        super(RandomRing, self).__init__(N=self.N, W=self.W, gtype=self.gtype,
                                         coords=self.coords, limits=self.limits)


class SwissRoll(Graph):
    r"""
    Creates a a swiss roll graph

    Parameters
    ----------
    N (int) : Number of vertices
        default 400
    s (float) : sigma
        default sqrt(2./N)
    thresh (float) : threshold
        default 1e-6
    rand_state : rand seed
        default 45

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.SwissRoll()

    """

    def __init__(self, N=400, a=1, b=4, dim=3, thresh=1e-6, s=None,
                 noise=False, srtype='uniform'):

        self.dim = dim
        self.N = N
        if s is None:
            s = sqrt(2./N)

        y1 = np.random.rand(N)
        y2 = np.random.rand(N)
        if srtype == 'uniform':
            tt = np.sqrt((b * b - a * a) * y1 + a * a)
        elif srtype == 'classic':
            tt = (b - a) * y1 + a
        self.gtype = 'swiss roll' + srtype
        tt *= pi
        h = 21 * y2
        if dim == 2:
            x = np.array((tt*np.cos(tt), tt * np.sin(tt)))
        elif dim == 3:
            x = np.array((tt*np.cos(tt), h, tt * np.sin(tt)))

        if noise:
            x += np.random.randn(*x.shape)

        self.x = x

        self.limits = np.array([-1, 1, -1, 1, -1, 1])

        coords = plotting.rescale_center(x)
        dist = utils.distanz(coords)
        W = np.exp(-np.power(dist, 2) / (2. * s**2))
        W -= np.diag(np.diag(W))
        W = np.where(W < thresh, 0, W)

        self.W = W

        self.coords = coords.transpose()
        super(SwissRoll, self).__init__(W=self.W, coords=self.coords,
                                        limits=self.limits, gtype=self.gtype)


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
