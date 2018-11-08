# -*- coding: utf-8 -*-

import traceback

import numpy as np
from scipy import sparse
import scipy.spatial as sps
import scipy.spatial.distance as spsd
import multiprocessing

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5

_logger = utils.build_logger(__name__)

# conversion between the FLANN conventions and the various backend functions
_dist_translation = {
                        'scipy-kdtree': {
                                'euclidean': 2,
                                'manhattan': 1,
                                'max_dist': np.inf
                        },
                        'scipy-ckdtree': {
                                'euclidean': 2,
                                'manhattan': 1,
                                'max_dist': np.inf
                        },
                        'scipy-pdist' : {
                                'euclidean': 'euclidean',
                                'manhattan': 'cityblock',
                                'max_dist': 'chebyshev',
                                'minkowski': 'minkowski'
                        },
                        'nmslib' : {
                                'euclidean': 'l2',
                                'manhattan': 'l1',
                                'max_dist': 'linf',
                                'minkowski': 'lp'
                        }
                    }

def _import_cfl():
    try:
        import cyflann as cfl
    except Exception:
        raise ImportError('Cannot import cyflann. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install cyflann.')
    return cfl

def _import_nmslib():
    try:
        import nmslib as nms
    except Exception:
        raise ImportError('Cannot import nmslib. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install nmslib.')
    return nms
    
def _knn_sp_kdtree(X, num_neighbors, dist_type, order=0):
    kdt = sps.KDTree(X)
    D, NN = kdt.query(X, k=(num_neighbors + 1), 
                      p=_dist_translation['scipy-kdtree'][dist_type])
    return NN, D

def _knn_sp_ckdtree(X, num_neighbors, dist_type, order=0):
    kdt = sps.cKDTree(X)
    D, NN = kdt.query(X, k=(num_neighbors + 1), 
                      p=_dist_translation['scipy-ckdtree'][dist_type], 
                      n_jobs=-1)
    return NN, D


def _knn_flann(X, num_neighbors, dist_type, order):
    # the combination FLANN + max_dist produces incorrect results
    # do not allow it
    if dist_type == 'max_dist':
        raise ValueError('FLANN and max_dist is not supported')
    
    cfl = _import_cfl()
    cfl.set_distance_type(dist_type, order=order)
    c = cfl.FLANNIndex(algorithm='kdtree')
    c.build_index(X)
    # Default FLANN parameters (I tried changing the algorithm and
    # testing performance on huge matrices, but the default one
    # seems to work best).
    NN, D = c.nn_index(X, num_neighbors + 1)
    c.free_index()
    if dist_type == 'euclidean': # flann returns squared distances
        return NN, np.sqrt(D)
    return NN, D

def _radius_sp_kdtree(X, epsilon, dist_type, order=0):
    kdt = sps.KDTree(X)
    D, NN = kdt.query(X, k=None, distance_upper_bound=epsilon,
                   p=_dist_translation['scipy-kdtree'][dist_type])
    return NN, D

def _radius_sp_ckdtree(X, epsilon, dist_type, order=0):
    N, dim = np.shape(X)
    kdt = sps.cKDTree(X)
    nn = kdt.query_ball_point(X, r=epsilon,
                              p=_dist_translation['scipy-ckdtree'][dist_type],
                              n_jobs=-1)
    D = []
    NN = []
    for k in range(N):
        x = np.matlib.repmat(X[k, :], len(nn[k]), 1)
        d = np.linalg.norm(x - X[nn[k], :], 
                           ord=_dist_translation['scipy-ckdtree'][dist_type],
                           axis=1)
        nidx = d.argsort()
        NN.append(np.take(nn[k], nidx))
        D.append(np.sort(d))
    return NN, D


def _knn_sp_pdist(X, num_neighbors, dist_type, order):
    if dist_type == 'minkowski':
        p = spsd.pdist(X, metric=_dist_translation['scipy-pdist'][dist_type], 
                        p=order)
    else:
        p = spsd.pdist(X, metric=_dist_translation['scipy-pdist'][dist_type])
    pd = spsd.squareform(p)
    pds = np.sort(pd)[:, 0:num_neighbors+1]
    pdi = pd.argsort()[:, 0:num_neighbors+1]
    return pdi, pds
    
def _knn_nmslib(X, num_neighbors, dist_type, order):
    N, dim = np.shape(X)
    ncpu = multiprocessing.cpu_count()
    if dist_type == 'minkowski':
        raise ValueError('unsupported distance type (lp) for nmslib')
    nms = _import_nmslib()
    nmsidx = nms.init(space=_dist_translation['nmslib'][dist_type])
    nmsidx.addDataPointBatch(X)
    nmsidx.createIndex()
    q = nmsidx.knnQueryBatch(X, k=num_neighbors+1, num_threads=int(ncpu/2))
    nn, d = zip(*q)
    D = np.concatenate(d).reshape(N, num_neighbors+1)
    NN = np.concatenate(nn).reshape(N, num_neighbors+1)
    return NN, D
    
def _radius_sp_pdist(X, epsilon, dist_type, order):
    N, dim = np.shape(X)
    if dist_type == 'minkowski':
        p = spsd.pdist(X, metric=_dist_translation['scipy-pdist'][dist_type], 
                       p=order)
    else:
        p = spsd.pdist(X, metric=_dist_translation['scipy-pdist'][dist_type])
    pd = spsd.squareform(p)
    pdf = pd < epsilon
    D = []
    NN = []
    for k in range(N):
        v = pd[k, pdf[k, :]]
        d = pd[k, :].argsort()
        # use the same conventions as in scipy.distance.kdtree
        NN.append(d[0:len(v)])
        D.append(np.sort(v))
    
    return NN, D

def _radius_flann(X, epsilon, dist_type, order=0):
    N, dim = np.shape(X)
    # the combination FLANN + max_dist produces incorrect results
    # do not allow it
    if dist_type == 'max_dist':
        raise ValueError('FLANN and max_dist is not supported')
    
    cfl = _import_cfl()
    cfl.set_distance_type(dist_type, order=order)
    c = cfl.FLANNIndex(algorithm='kdtree')
    c.build_index(X)
    D = []
    NN = []
    for k in range(N):
        nn, d = c.nn_radius(X[k, :], epsilon*epsilon)
        D.append(d)
        NN.append(nn)
    c.free_index()
    if dist_type == 'euclidean': # flann returns squared distances
        return NN, list(map(np.sqrt, D))
    return NN, D

def _radius_nmslib(X, epsilon, dist_type, order=0):
    raise ValueError('nmslib does not support (yet?) range queries')

def center_input(X, N):
    return X - np.kron(np.ones((N, 1)), np.mean(X, axis=0))

def rescale_input(X, N, d):
     bounding_radius = 0.5 * np.linalg.norm(np.amax(X, axis=0) -
                                                   np.amin(X, axis=0), 2)
     scale = np.power(N, 1. / float(min(d, 3))) / 10.
     return X * scale / bounding_radius

class NNGraph(Graph):
    r"""Nearest-neighbor graph from given point cloud.

    Parameters
    ----------
    Xin : ndarray
        Input points, Should be an `N`-by-`d` matrix, where `N` is the number
        of nodes in the graph and `d` is the dimension of the feature space.
    NNtype : string, optional
        Type of nearest neighbor graph to create. The options are 'knn' for
        k-Nearest Neighbors or 'radius' for epsilon-Nearest Neighbors (default
        is 'knn').
    backend : {'scipy-kdtree', 'scipy-pdist', 'flann'}
        Type of the backend for graph construction. 
        - 'scipy-kdtree' will use scipy.spatial.KDTree
        - 'scipy-ckdtree'(default) will use scipy.spatial.cKDTree
        - 'scipy-pdist' will use scipy.spatial.distance.pdist (slowest but exact)
        - 'flann' use Fast Library for Approximate Nearest Neighbors (FLANN)
        - 'nmslib' use nmslib for approximate nearest neighbors (faster in high-dimensional spaces)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width parameter of the similarity kernel (default is 0.1)
    epsilon : float, optional
        Radius for the epsilon-neighborhood search (default is 0.01)
    plotting : dict, optional
        Dictionary of plotting parameters. See :obj:`pygsp.plotting`.
        (default is {})
    symmetrize_type : string, optional
        Type of symmetrization to use for the adjacency matrix. See
        :func:`pygsp.utils.symmetrization` for the options.
        (default is 'average')
    dist_type : string, optional
        Type of distance to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
        (default is 'euclidean')
    order : float, optional
        Only used if dist_type is 'minkowski'; represents the order of the
        Minkowski distance. (default is 0)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.RandomState(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(X)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    """


    def __init__(self, Xin, NNtype='knn', backend='scipy-ckdtree', center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01,
                 plotting={}, symmetrize_type='average', dist_type='euclidean',
                 order=0, **kwargs):

        self.Xin = Xin
        self.NNtype = NNtype
        self.backend = backend
        self.center = center
        self.rescale = rescale
        self.k = k
        self.sigma = sigma
        self.epsilon = epsilon

        _dist_translation['scipy-kdtree']['minkowski'] = order
        _dist_translation['scipy-ckdtree']['minkowski'] = order

        self._nn_functions = {
                'knn': {
                        'scipy-kdtree': _knn_sp_kdtree,
                        'scipy-ckdtree': _knn_sp_ckdtree,
                        'scipy-pdist': _knn_sp_pdist,
                        'flann': _knn_flann,
                        'nmslib': _knn_nmslib
                        },
                'radius': {
                        'scipy-kdtree': _radius_sp_kdtree,
                        'scipy-ckdtree': _radius_sp_ckdtree,
                        'scipy-pdist': _radius_sp_pdist,
                        'flann': _radius_flann,
                        'nmslib': _radius_nmslib
                        },
                } 
        
       

        self.symmetrize_type = symmetrize_type
        self.dist_type = dist_type
        self.order = order

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if self.center:
            Xout = center_input(Xout, N)

        if self.rescale:
            Xout = rescale_input(Xout, N, d)

       
        if self._nn_functions.get(NNtype) == None:
            raise ValueError('Invalid NNtype {}'.format(self.NNtype))

        if self._nn_functions[NNtype].get(backend) == None:
            raise ValueError('Invalid backend {} for type {}'.format(backend, 
                             self.NNtype))
        if self.NNtype == 'knn':
            NN, D = self._nn_functions[NNtype][backend](Xout, k, 
                                                        dist_type, order)
            
        elif self.NNtype == 'radius':
            NN, D = self._nn_functions[NNtype][backend](Xout, epsilon, 
                                                         dist_type, order)
        countV = list(map(len, NN))
        count = sum(countV)    
        spi = np.zeros((count))
        spj = np.zeros((count))
        spv = np.zeros((count))

        start = 0
        for i in range(N):
            leng = countV[i] - 1
            spi[start:start + leng] = np.kron(np.ones((leng)), i)
            spj[start:start + leng] = NN[i][1:]
            spv[start:start + leng] = np.exp(-np.power(D[i][1:], 2) /
                                       float(self.sigma))
            start = start + leng


        W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError('Weight matrix W is not square')

        # Enforce symmetry. Note that checking symmetry with
        # np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method=symmetrize_type)

        super(NNGraph, self).__init__(W=W, plotting=plotting,
                                      coords=Xout, **kwargs)

    def _get_extra_repr(self):
        return {'NNtype': self.NNtype,
                'backend': self.backend,
                'center': self.center,
                'rescale': self.rescale,
                'k': self.k,
                'sigma': '{:.2f}'.format(self.sigma),
                'epsilon': '{:.2f}'.format(self.epsilon),
                'symmetrize_type': self.symmetrize_type,
                'dist_type': self.dist_type,
                'order': self.order}
