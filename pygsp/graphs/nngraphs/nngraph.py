# -*- coding: utf-8 -*-

import traceback

import numpy as np
from scipy import sparse, spatial

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
                        'scipy-pdist' : {
                                'euclidean': 'euclidean',
                                'manhattan': 'cityblock',
                                'max_dist': 'chebyshev',
                                'minkowski': 'minkowski'
                        },
                        
                    }

def _import_pfl():
    try:
        import pyflann as pfl
    except Exception as e:
        raise ImportError('Cannot import pyflann. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install pyflann (or pyflann3). '
                          'Original exception: {}'.format(e))
    return pfl

    
    
def _knn_sp_kdtree(X, num_neighbors, dist_type, order=0):
    kdt = spatial.KDTree(X)
    D, NN = kdt.query(X, k=(num_neighbors + 1), 
                      p=_dist_translation['scipy-kdtree'][dist_type])
    return NN, D

def _knn_flann(X, num_neighbors, dist_type, order):
    pfl = _import_pfl()
    # the combination FLANN + max_dist produces incorrect results
    # do not allow it
    if dist_type == 'max_dist':
        raise ValueError('FLANN and max_dist is not supported')
    pfl.set_distance_type(dist_type, order=order)
    flann = pfl.FLANN()

    # Default FLANN parameters (I tried changing the algorithm and
    # testing performance on huge matrices, but the default one
    # seems to work best).
    NN, D = flann.nn(X, X, num_neighbors=(num_neighbors + 1), 
                     algorithm='kdtree')
    return NN, D

def _radius_sp_kdtree(X, epsilon, dist_type, order=0):
    kdt = spatial.KDTree(X)
    D, NN = kdt.query(X, k=None, distance_upper_bound=epsilon,
                              p=_dist_translation['scipy-kdtree'][dist_type])
    return NN, D

def _knn_sp_pdist(X, num_neighbors, dist_type, _order):
    pd = spatial.distance.squareform(
            spatial.distance.pdist(X, 
                                   _dist_translation['scipy-pdist'][dist_type], 
                                   p=_order))
    pds = np.sort(pd)[:, 0:num_neighbors+1]
    pdi = pd.argsort()[:, 0:num_neighbors+1]
    return pdi, pds
    
def _radius_sp_pdist(_X, _epsilon, _dist_type, order=0):
    raise NotImplementedError()

def _radius_flann(_X, _epsilon, _dist_type, order=0):
    raise NotImplementedError()

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
        - 'scipy-kdtree'(default) will use scipy.spatial.KDTree
        - 'scipy-pdist' will use scipy.spatial.distance.pdist (slowest but exact)
        - 'flann' use Fast Library for Approximate Nearest Neighbors (FLANN)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width of the similarity kernel.
        By default, it is set to the average of the nearest neighbor distance.
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


    def __init__(self, Xin, NNtype='knn', backend='scipy-kdtree', center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01, gtype=None,
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

        self._nn_functions = {
                'knn': {
                        'scipy-kdtree': _knn_sp_kdtree,
                        'scipy-pdist': _knn_sp_pdist,
                        'flann': _knn_flann
                        },
                'radius': {
                        'scipy-kdtree': _radius_sp_kdtree,
                        'scipy-pdist': _radius_sp_pdist,
                        'flann': _radius_flann
                        },
                } 
        
        if gtype is None:
            gtype = 'nearest neighbors'
        else:
            gtype = '{}, NNGraph'.format(gtype)

        self.symmetrize_type = symmetrize_type
        self.dist_type = dist_type
        self.order = order

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if k >= N:
            raise ValueError('The number of neighbors (k={}) must be smaller '
                             'than the number of nodes ({}).'.format(k, N))

        if self.center:
            Xout = self.Xin - np.kron(np.ones((N, 1)),
                                      np.mean(self.Xin, axis=0))

        if self.rescale:
            bounding_radius = 0.5 * np.linalg.norm(np.amax(Xout, axis=0) -
                                                   np.amin(Xout, axis=0), 2)
            scale = np.power(N, 1. / float(min(d, 3))) / 10.
            Xout *= scale / bounding_radius

       

        if self.NNtype == 'knn':
            spi = np.zeros((N * k))
            spj = np.zeros((N * k))
            spv = np.zeros((N * k))

            NN, D = self._nn_functions[NNtype][backend](Xout, k, 
                                                        dist_type, order)

            if self.sigma is None:
                self.sigma = np.mean(D[:, 1:])  # Discard distance to self.

            for i in range(N):
                spi[i * k:(i + 1) * k] = np.kron(np.ones((k)), i)
                spj[i * k:(i + 1) * k] = NN[i, 1:]
                spv[i * k:(i + 1) * k] = np.exp(-np.power(D[i, 1:], 2) /
                                                float(self.sigma))

        elif self.NNtype == 'radius':

            NN, D = self._nn_functions[NNtype][backend](Xout, epsilon, 

                                                         dist_type, order)

            count = sum(map(len, NN))
            

            spi = np.zeros((count))
            spj = np.zeros((count))
            spv = np.zeros((count))

            start = 0
            for i in range(N):
                leng = len(NN[i]) - 1
                spi[start:start + leng] = np.kron(np.ones((leng)), i)
                spj[start:start + leng] = NN[i][1:]
                spv[start:start + leng] = np.exp(-np.power(D[i][1:], 2) /
                                                 float(self.sigma))
                start = start + leng

        else:
            raise ValueError('Unknown NNtype {}'.format(self.NNtype))

        W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError('Weight matrix W is not square')

        # Enforce symmetry. Note that checking symmetry with
        # np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method=symmetrize_type)

        super(NNGraph, self).__init__(W, plotting=plotting,
                                      coords=Xout, **kwargs)

    def _get_extra_repr(self):
        return {'NNtype': self.NNtype,
                'use_flann': self.use_flann,
                'center': self.center,
                'rescale': self.rescale,
                'k': self.k,
                'sigma': '{:.2f}'.format(self.sigma),
                'epsilon': '{:.2f}'.format(self.epsilon),
                'symmetrize_type': self.symmetrize_type,
                'dist_type': self.dist_type,
                'order': self.order}
