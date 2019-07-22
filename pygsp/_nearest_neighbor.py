# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import sparse, spatial
from pygsp import utils

def _scipy_pdist(features, metric, order, kind, k, radius, params):
    if params:
        raise ValueError('unexpected parameters {}'.format(params))
    metric = 'cityblock' if metric == 'manhattan' else metric
    metric = 'chebyshev' if metric == 'max_dist' else metric
    params = dict(metric=metric)
    if metric == 'minkowski':
        params['p'] = order
    dist = spatial.distance.pdist(features, **params)
    dist = spatial.distance.squareform(dist)
    if kind == 'knn':
        neighbors = np.argsort(dist)[:, :k+1]
        distances = np.take_along_axis(dist, neighbors, axis=-1)
    elif kind == 'radius':
        distances = []
        neighbors = []
        for distance in dist:
            neighbor = np.flatnonzero(distance < radius)
            neighbors.append(neighbor)
            distances.append(distance[neighbor])
    return neighbors, distances


def _scipy_kdtree(features, _, order, kind, k, radius, params):
    if order is None:
        raise ValueError('invalid metric for scipy-kdtree')
    eps = params.pop('eps', 0)
    tree = spatial.KDTree(features, **params)
    params = dict(p=order, eps=eps)
    if kind == 'knn':
        params['k'] = k + 1
    elif kind == 'radius':
        params['k'] = None
        params['distance_upper_bound'] = radius
    distances, neighbors = tree.query(features, **params)
    return neighbors, distances


def _scipy_ckdtree(features, _, order, kind, k, radius, params):
    if order is None:
        raise ValueError('invalid metric for scipy-kdtree')
    eps = params.pop('eps', 0)
    tree = spatial.cKDTree(features, **params)
    params = dict(p=order, eps=eps, n_jobs=-1)
    if kind == 'knn':
        params['k'] = k + 1
    elif kind == 'radius':
        params['k'] = features.shape[0]  # number of vertices
        params['distance_upper_bound'] = radius
    distances, neighbors = tree.query(features, **params)
    if kind == 'knn':
        return neighbors, distances
    elif kind == 'radius':
        dist = []
        neigh = []
        for distance, neighbor in zip(distances, neighbors):
            mask = (distance != np.inf)
            dist.append(distance[mask])
            neigh.append(neighbor[mask])
        return neigh, dist


def _flann(features, metric, order, kind, k, radius, params):
    if metric == 'max_dist':
        raise ValueError('flann gives wrong results for metric="max_dist".')
    try:
        import cyflann as cfl
    except Exception as e:
        raise ImportError('Cannot import cyflann. Choose another nearest '
                          'neighbors backend or try to install it with '
                          'pip (or conda) install cyflann. '
                          'Original exception: {}'.format(e))
    cfl.set_distance_type(metric, order=order)
    index = cfl.FLANNIndex()
    index.build_index(features, **params)
    # I tried changing the algorithm and testing performance on huge matrices,
    # but the default parameters seems to work best.
    if kind == 'knn':
        neighbors, distances = index.nn_index(features, k+1)
        if metric == 'euclidean':
            np.sqrt(distances, out=distances)
        elif metric == 'minkowski':
            np.power(distances, 1/order, out=distances)
    elif kind == 'radius':
        distances = []
        neighbors = []
        if metric == 'euclidean':
            radius = radius**2
        elif metric == 'minkowski':
            radius = radius**order
        n_vertices, _ = features.shape
        for vertex in range(n_vertices):
            neighbor, distance = index.nn_radius(features[vertex, :], radius)
            distances.append(distance)
            neighbors.append(neighbor)
        if metric == 'euclidean':
            distances = list(map(np.sqrt, distances))
        elif metric == 'minkowski':
            distances = list(map(lambda d: np.power(d, 1/order), distances))
    index.free_index()
    return neighbors, distances


def _nmslib(features, metric, order, kind, k, _, params):
    if kind == 'radius':
        raise ValueError('nmslib does not support kind="radius".')
    if metric == 'minkowski':
        raise ValueError('nmslib does not support metric="minkowski".')
    try:
        import nmslib as nms
    except Exception as e:
        raise ImportError('Cannot import nmslib. Choose another nearest '
                          'neighbors backend or try to install it with '
                          'pip (or conda) install nmslib. '
                          'Original exception: {}'.format(e))
    n_vertices, _ = features.shape
    params_index = params.pop('index', None)
    params_query = params.pop('query', None)
    metric = 'l2' if metric == 'euclidean' else metric
    metric = 'l1' if metric == 'manhattan' else metric
    metric = 'linf' if metric == 'max_dist' else metric
    index = nms.init(space=metric, **params)
    index.addDataPointBatch(features)
    index.createIndex(params_index)
    if params_query is not None:
        index.setQueryTimeParams(params_query)
    results = index.knnQueryBatch(features, k=k+1)
    neighbors, distances = zip(*results)
    distances = np.concatenate(distances).reshape(n_vertices, k+1)
    neighbors = np.concatenate(neighbors).reshape(n_vertices, k+1)
    return neighbors, distances

def nn(features, metric='euclidean', order=2, kind='knn', k=10, radius=None, backend='scipy-ckdtree', **kwargs):
    '''Find nearest neighboors.
    
    Parameters
    ----------
    features : data numpy array 
    metric : {'euclidean', 'manhattan', 'minkowski', 'max_dist'}, optional
        Metric used to compute pairwise distances.

        * ``'euclidean'`` defines pairwise distances as
          :math:`d(v_i, v_j) = \| x_i - x_j \|_2`.
        * ``'manhattan'`` defines pairwise distances as
          :math:`d(v_i, v_j) = \| x_i - x_j \|_1`.
        * ``'minkowski'`` generalizes the above and defines distances as
          :math:`d(v_i, v_j) = \| x_i - x_j \|_p`
          where :math:`p` is the ``order`` of the norm.
        * ``'max_dist'`` defines pairwise distances as
          :math:`d(v_i, v_j) = \| x_i - x_j \|_\infty = \max(x_i - x_j)`, where
          the maximum is taken over the elements of the vector.

        More metrics may be supported for some backends.
        Please refer to the documentation of the chosen backend.
    kind : 'knn' or 'radius' (default 'knn')
    k : number of nearest neighboors if 'knn' is selected
    radius : radius of the search if 'radius' is slected
    
    order : float, optional
        The order of the Minkowski distance for ``metric='minkowski'``.
    backend : string, optional
        * ``'scipy-pdist'`` uses :func:`scipy.spatial.distance.pdist` to
          compute pairwise distances. The method is brute force and computes
          all distances. That is the slowest method.
        * ``'scipy-kdtree'`` uses :class:`scipy.spatial.KDTree`. The method
          builds a k-d tree to prune the number of pairwise distances it has to
          compute. That is an efficient strategy for low-dimensional spaces.
        * ``'scipy-ckdtree'`` uses :class:`scipy.spatial.cKDTree`. The same as
          ``'scipy-kdtree'`` but with C bindings, which should be faster.
          That is the default.
        * ``'flann'`` uses the `Fast Library for Approximate Nearest Neighbors
          (FLANN) <https://github.com/mariusmuja/flann>`_. That method is an
          approximation.
        * ``'nmslib'`` uses the `Non-Metric Space Library (NMSLIB)
          <https://github.com/nmslib/nmslib>`_. That method is an
          approximation. It should be the fastest in high-dimensional spaces.

        You can look at this `benchmark
        <https://github.com/erikbern/ann-benchmarks>`_ to get an idea of the
        relative performance of those backends. It's nonetheless wise to run
        some tests on your own data.   
    '''
    if kind=='knn':
        radius = None
    elif kind=='radius':
        k = None
    else:
        raise ValueError('"kind" must be "knn" or "radius"')
    
    _orders = {
        'euclidean': 2,
        'manhattan': 1,
        'max_dist': np.inf,
        'minkowski': order,
    }
    order = _orders.pop(metric, None)  
    try:
        function = globals()['_' + backend.replace('-', '_')]
    except KeyError:
        raise ValueError('Invalid backend "{}".'.format(backend))
    neighbors, distances = function(features, metric, order,
                                    kind, k, radius, kwargs)
    return neighbors, distances


def sparse_distance_matrix(neighbors, distances, symmetrize=True, safe=False, kind = None):
    '''Build a sparse distance matrix.'''
    n_edges = [len(n) - 1 for n in neighbors]  # remove distance to self
    if safe and kind is None:
        raise ValueError('Please specify "kind" to "knn" or "radius" to use the safe mode')
    
    if safe and kind == 'radius':
        n_disconnected = np.sum(np.asarray(n_edges) == 0)
        if n_disconnected > 0:
            _logger.warning('{} points (out of {}) have no neighboors. '
                            'Consider increasing the radius or setting '
                            'kind=knn.'.format(n_disconnected, n_vertices))

    value = np.empty(sum(n_edges), dtype=np.float)
    row = np.empty_like(value, dtype=np.int)
    col = np.empty_like(value, dtype=np.int)
    start = 0
    n_vertices = len(n_edges)
    for vertex in range(n_vertices):
        if safe and kind == 'knn':
            assert n_edges[vertex] == k
        end = start + n_edges[vertex]
        value[start:end] = distances[vertex][1:]
        row[start:end] = np.full(n_edges[vertex], vertex)
        col[start:end] = neighbors[vertex][1:]
        start = end
    W = sparse.csr_matrix((value, (row, col)), (n_vertices, n_vertices))
    if symmetrize:
        # Enforce symmetry. May have been broken by k-NN. Checking symmetry
        # with np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method='fill')
    return W