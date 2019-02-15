# -*- coding: utf-8 -*-

from __future__ import division

import multiprocessing
import traceback

import numpy as np
from scipy import sparse, spatial

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5


# conversion between the FLANN conventions and the various backend functions
_metrics = {
    'scipy-pdist': {
        'euclidean': 'euclidean',
        'manhattan': 'cityblock',
        'max_dist': 'chebyshev',
        'minkowski': 'minkowski',
    },
    'scipy-kdtree': {
        'euclidean': 2,
        'manhattan': 1,
        'max_dist': np.inf,
        'minkowski': 0,
    },
    'scipy-ckdtree': {
        'euclidean': 2,
        'manhattan': 1,
        'max_dist': np.inf,
        'minkowski': 0,
    },
    'flann': {
        'euclidean': 'euclidean',
        'manhattan': 'manhattan',
#        'max_dist': 'max_dist',  # produces incorrect results
        'minkowski': 'minkowski',
    },
    'nmslib': {
        'euclidean': 'l2',
        'manhattan': 'l1',
        'max_dist': 'linf',
#        'minkowski': 'lp',  # unsupported
    }
}


def _import_cfl():
    try:
        import cyflann as cfl
    except Exception as e:
        raise ImportError('Cannot import cyflann. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install cyflann. '
                          'Original exception: {}'.format(e))
    return cfl


def _import_nmslib():
    try:
        import nmslib as nms
    except Exception:
        raise ImportError('Cannot import nmslib. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install nmslib.')
    return nms


def _knn_sp_kdtree(features, num_neighbors, metric, order):
    p = order if metric == 'minkowski' else _metrics['scipy-kdtree'][metric]
    kdt = spatial.KDTree(features)
    D, NN = kdt.query(features, k=(num_neighbors + 1), p=p)
    return NN, D


def _knn_sp_ckdtree(features, num_neighbors, metric, order):
    p = order if metric == 'minkowski' else _metrics['scipy-ckdtree'][metric]
    kdt = spatial.cKDTree(features)
    D, NN = kdt.query(features, k=(num_neighbors + 1), p=p, n_jobs=-1)
    return NN, D


def _knn_flann(features, num_neighbors, metric, order):
    cfl = _import_cfl()
    cfl.set_distance_type(metric, order=order)
    c = cfl.FLANNIndex(algorithm='kdtree')
    c.build_index(features)
    # Default FLANN parameters (I tried changing the algorithm and
    # testing performance on huge matrices, but the default one
    # seems to work best).
    NN, D = c.nn_index(features, num_neighbors + 1)
    c.free_index()
    if metric == 'euclidean':  # flann returns squared distances
        return NN, np.sqrt(D)
    return NN, D


def _radius_sp_kdtree(features, radius, metric, order):
    p = order if metric == 'minkowski' else _metrics['scipy-kdtree'][metric]
    kdt = spatial.KDTree(features)
    D, NN = kdt.query(features, k=None, distance_upper_bound=radius, p=p)
    return NN, D


def _radius_sp_ckdtree(features, radius, metric, order):
    p = order if metric == 'minkowski' else _metrics['scipy-ckdtree'][metric]
    n_vertices, _ = features.shape
    kdt = spatial.cKDTree(features)
    nn = kdt.query_ball_point(features, r=radius, p=p, n_jobs=-1)
    D = []
    NN = []
    for k in range(n_vertices):
        x = np.tile(features[k, :], (len(nn[k]), 1))
        d = np.linalg.norm(x - features[nn[k], :],
                           ord=_metrics['scipy-ckdtree'][metric],
                           axis=1)
        nidx = d.argsort()
        NN.append(np.take(nn[k], nidx))
        D.append(np.sort(d))
    return NN, D


def _knn_sp_pdist(features, num_neighbors, metric, order):
    if metric == 'minkowski':
        p = spatial.distance.pdist(features,
                                   metric=_metrics['scipy-pdist'][metric],
                                   p=order)
    else:
        p = spatial.distance.pdist(features,
                                   metric=_metrics['scipy-pdist'][metric])
    pd = spatial.distance.squareform(p)
    pds = np.sort(pd)[:, :num_neighbors+1]
    pdi = pd.argsort()[:, :num_neighbors+1]
    return pdi, pds


def _knn_nmslib(features, num_neighbors, metric, _):
    n_vertices, _ = features.shape
    ncpu = multiprocessing.cpu_count()
    nms = _import_nmslib()
    nmsidx = nms.init(space=_metrics['nmslib'][metric])
    nmsidx.addDataPointBatch(features)
    nmsidx.createIndex()
    q = nmsidx.knnQueryBatch(features, k=num_neighbors+1,
                             num_threads=int(ncpu/2))
    nn, d = zip(*q)
    D = np.concatenate(d).reshape(n_vertices, num_neighbors+1)
    NN = np.concatenate(nn).reshape(n_vertices, num_neighbors+1)
    return NN, D


def _radius_sp_pdist(features, radius, metric, order):
    n_vertices, _ = features.shape
    if metric == 'minkowski':
        p = spatial.distance.pdist(features,
                                   metric=_metrics['scipy-pdist'][metric],
                                   p=order)
    else:
        p = spatial.distance.pdist(features,
                                   metric=_metrics['scipy-pdist'][metric])
    pd = spatial.distance.squareform(p)
    pdf = pd < radius
    D = []
    NN = []
    for k in range(n_vertices):
        v = pd[k, pdf[k, :]]
        d = pd[k, :].argsort()
        # use the same conventions as in scipy.distance.kdtree
        NN.append(d[0:len(v)])
        D.append(np.sort(v))
    return NN, D


def _radius_flann(features, radius, metric, order):
    n_vertices, _ = features.shape
    cfl = _import_cfl()
    cfl.set_distance_type(metric, order=order)
    c = cfl.FLANNIndex(algorithm='kdtree')
    c.build_index(features)
    D = []
    NN = []
    for k in range(n_vertices):
        nn, d = c.nn_radius(features[k, :], radius**2)
        D.append(d)
        NN.append(nn)
    c.free_index()
    if metric == 'euclidean':
        # Flann returns squared distances.
        D = list(map(np.sqrt, D))
    return NN, D


_nn_functions = {
    'knn': {
        'scipy-pdist': _knn_sp_pdist,
        'scipy-kdtree': _knn_sp_kdtree,
        'scipy-ckdtree': _knn_sp_ckdtree,
        'flann': _knn_flann,
        'nmslib': _knn_nmslib,
    },
    'radius': {
        'scipy-pdist': _radius_sp_pdist,
        'scipy-kdtree': _radius_sp_kdtree,
        'scipy-ckdtree': _radius_sp_ckdtree,
        'flann': _radius_flann,
    },
}


def center_features(features):
    n_vertices, _ = features.shape
    return features - np.kron(np.ones((n_vertices, 1)), np.mean(features, 0))


def rescale_features(features):
    n_vertices, dimensionality = features.shape
    bounding_radius = 0.5 * np.linalg.norm(np.amax(features, axis=0) -
                                           np.amin(features, axis=0), 2)
    scale = np.power(n_vertices, 1 / min(dimensionality, 3)) / 10
    return features * scale / bounding_radius


class NNGraph(Graph):
    r"""Nearest-neighbor graph.

    The nearest-neighbor graph is built from a set of features, where the edge
    weight between vertices :math:`v_i` and :math:`v_j` is given by

    .. math:: A(i,j) = \exp \left( -\frac{d^2(v_i, v_j)}{\sigma^2} \right),

    where :math:`d(v_i, v_j)` is a distance measure between some representation
    (the features) of :math:`v_i` and :math:`v_j`. For example, the features
    might be the 3D coordinates of points in a point cloud. Then, if
    ``metric='euclidean'``, :math:`d(v_i, v_j) = \| x_i - x_j \|_2`, where
    :math:`x_i` is the 3D position of vertex :math:`v_i`.

    The similarity matrix :math:`A` is sparsified by either keeping the ``k``
    closest vertices for each vertex (if ``type='knn'``), or by setting to zero
    any distance greater than ``radius`` (if ``type='radius'``).

    Parameters
    ----------
    features : ndarray
        An `N`-by-`d` matrix, where `N` is the number of nodes in the graph and
        `d` is the number of features.
    center : bool, optional
        Whether to center the features to have zero mean.
    rescale : bool, optional
        Whether to scale the features so that they lie in an l2-ball.
    metric : {'euclidean', 'manhattan', 'minkowski', 'max_dist'}, optional
        Metric used to compute pairwise distances.
    order : float, optional
        The order of the Minkowski distance for ``metric='minkowski'``.
    kind : {'knn', 'radius'}, optional
        Kind of nearest neighbor graph to create. Either ``'knn'`` for
        k-nearest neighbors or ``'radius'`` for epsilon-nearest neighbors.
    k : int, optional
        Number of neighbors considered when building a k-NN graph with
        ``type='knn'``.
    radius : float, optional
        Radius of the ball when building a radius graph with ``type='radius'``.
    kernel_width : float, optional
        Width of the Gaussian kernel. By default, it is set to the average of
        the distances of neighboring vertices.
    backend : string, optional
        * ``'scipy-pdist'`` uses :func:`scipy.spatial.distance.pdist` to
          compute pairwise distances. The method is brute force and computes
          all distances. That is the slowest method.
        * ``'scipy-kdtree'`` uses :class:`scipy.spatial.KDTree`. The method
          builds a k-d tree to prune the number of pairwise distances it has to
          compute.
        * ``'scipy-ckdtree'`` uses :class:`scipy.spatial.cKDTree`. The same as
          ``'scipy-kdtree'`` but with C bindings, which should be faster.
        * ``'flann'`` uses the `Fast Library for Approximate Nearest Neighbors
          (FLANN) <https://github.com/mariusmuja/flann>`_. That method is an
          approximation.
        * ``'nmslib'`` uses the `Non-Metric Space Library (NMSLIB)
          <https://github.com/nmslib/nmslib>`_. That method is an
          approximation. It should be the fastest in high-dimensional spaces.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> features = np.random.RandomState(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(features)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, features, center=True, rescale=True,
                 metric='euclidean', order=3,
                 kind='knn', k=10, radius=0.01,
                 kernel_width=None,
                 backend='scipy-ckdtree',
                 **kwargs):

        self.features = features  # stored in coords, but scaled and centered
        self.center = center
        self.rescale = rescale
        self.metric = metric
        self.order = order
        self.kind = kind
        self.k = k
        self.radius = radius
        self.kernel_width = kernel_width
        self.backend = backend

        N, d = np.shape(features)

        if _nn_functions.get(kind) is None:
            raise ValueError('Invalid kind "{}".'.format(kind))

        if backend not in _metrics.keys():
            raise ValueError('Invalid backend "{}".'.format(backend))

        if _metrics['scipy-pdist'].get(metric) is None:
            raise ValueError('Invalid metric "{}".'.format(metric))

        if _nn_functions[kind].get(backend) is None:
            raise ValueError('{} does not support kind "{}".'.format(
                backend, kind))

        if _metrics[backend].get(metric) is None:
            raise ValueError('{} does not support the {} metric.'.format(
                backend, metric))

        if kind == 'knn' and k >= N:
            raise ValueError('The number of neighbors (k={}) must be smaller '
                             'than the number of vertices ({}).'.format(k, N))

        if kind == 'radius' and radius <= 0:
            raise ValueError('The radius must be greater than 0.')

        if center:
            features = center_features(features)

        if rescale:
            features = rescale_features(features)

        if kind == 'knn':
            NN, D = _nn_functions[kind][backend](features, k, metric, order)
            if self.kernel_width is None:
                # Discard distance to self.
                self.kernel_width = np.mean(D[:, 1:])

        elif kind == 'radius':
            NN, D = _nn_functions[kind][backend](features, radius,
                                                 metric, order)
            if self.kernel_width is None:
                # Discard distance to self and deal with disconnected vertices.
                means = []
                for distance in D:
                    if len(distance) > 1:
                        means.append(np.mean(distance[1:]))
                self.kernel_width = np.mean(means) if len(means) > 0 else 0

        n_edges = [len(x) - 1 for x in NN]  # remove distance to self
        value = np.empty(sum(n_edges), dtype=np.float)
        row = np.empty_like(value, dtype=np.int)
        col = np.empty_like(value, dtype=np.int)

        if kind == 'radius':
            n_disconnected = np.sum(np.asarray(n_edges) == 0)
            if n_disconnected > 0:
                logger = utils.build_logger(__name__)
                logger.warning('{} vertices (out of {}) are disconnected. '
                               'Consider increasing the radius or setting '
                               'kind=knn.'.format(n_disconnected, N))

        start = 0
        for vertex in range(N):
            if kind == 'knn':
                assert n_edges[vertex] == k
            end = start + n_edges[vertex]
            distance = np.power(D[vertex][1:], 2)
            value[start:end] = np.exp(-distance / self.kernel_width)
            row[start:end] = np.full(n_edges[vertex], vertex)
            col[start:end] = NN[vertex][1:]
            start = end

        W = sparse.csc_matrix((value, (row, col)), shape=(N, N))

        # Enforce symmetry. May have been broken by k-NN. Checking symmetry
        # with np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method='average')

        super(NNGraph, self).__init__(W=W, coords=features, **kwargs)

    def _get_extra_repr(self):
        return {
            'center': self.center,
            'rescale': self.rescale,
            'metric': self.metric,
            'order': self.order,
            'kind': self.kind,
            'k': self.k,
            'radius': '{:.2f}'.format(self.radius),
            'kernel_width': '{:.2f}'.format(self.kernel_width),
            'backend': self.backend,
        }
