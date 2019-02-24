# -*- coding: utf-8 -*-

from __future__ import division

from functools import partial

import numpy as np
from scipy import sparse, spatial

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5


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
    except Exception:
        raise ImportError('Cannot import nmslib. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install nmslib.')
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


class NNGraph(Graph):
    r"""Nearest-neighbor graph.

    The nearest-neighbor graph is built from a set of features, where the edge
    weight between vertices :math:`v_i` and :math:`v_j` is given by

    .. math:: A(i,j) = k \left( \frac{d(v_i, v_j)}{\sigma} \right),

    where :math:`d(v_i, v_j)` is a distance measure between some representation
    (the features) of :math:`v_i` and :math:`v_j`, :math:`k` is a kernel
    function that transforms a distance in :math:`[0, \infty]` to a similarity
    measure in :math:`[0, 1]`, and :math:`\sigma` is the kernel width.

    For example, the features might be the 3D coordinates of points in a point
    cloud. Then, if ``metric='euclidean'`` and ``kernel='gaussian'`` (the
    defaults), :math:`A(i,j) = \exp(-\log(2) \| x_i - x_j \|_2^2 / \sigma^2)`,
    where :math:`x_i` is the 3D position of vertex :math:`v_i`.

    The similarity matrix :math:`A` is sparsified by either keeping the ``k``
    closest vertices for each vertex (if ``type='knn'``), or by setting to zero
    any distance greater than ``radius`` (if ``type='radius'``).

    Parameters
    ----------
    features : ndarray
        An `N`-by-`d` matrix, where `N` is the number of nodes in the graph and
        `d` is the number of features.
    standardize : bool, optional
        Whether to rescale the features so that each feature has a mean of 0
        and standard deviation of 1 (unit variance).
    metric : {'euclidean', 'manhattan', 'minkowski', 'max_dist'}, optional
        Metric used to compute pairwise distances.
        More metrics may be supported for some backends.
        Please refer to the documentation of the chosen backend.
    order : float, optional
        The order of the Minkowski distance for ``metric='minkowski'``.
    kind : {'knn', 'radius'}, optional
        Kind of nearest neighbor graph to create. Either ``'knn'`` for
        k-nearest neighbors or ``'radius'`` for epsilon-nearest neighbors.
    k : int, optional
        Number of neighbors considered when building a k-NN graph with
        ``type='knn'``.
    radius : float or {'estimate', 'estimate-knn'}, optional
        Radius of the ball when building a radius graph with ``type='radius'``.
        It is hard to set an optimal radius. If too small, some vertices won't
        be connected to any other vertex. If too high, vertices will be
        connected to many other vertices and the graph won't be sparse (high
        average degree).  If no good radius is known a priori, we can estimate
        one. ``'estimate'`` sets the radius as the expected average distance
        between vertices for a uniform sampling of the ambient space.
        ``'estimate-knn'`` first builds a knn graph and sets the radius to the
        average distance. ``'estimate-knn'`` usually gives a better estimation
        but is more costly. ``'estimate'`` can be better in low dimension.
    kernel : string or function
        The function :math:`k` that transforms a distance to a similarity.
        The following kernels are pre-defined.

        * ``'gaussian'`` defines the Gaussian, also known as the radial basis
          function (RBF), kernel :math:`k(d) = \exp(-\log(2) d^2)`.
        * ``'exponential'`` defines the kernel :math:`k(d) = \exp(-\log(2) d)`.
        * ``'rectangular'`` returns 1 if :math:`d < 1` and 0 otherwise.
        * ``'triangular'`` defines the kernel :math:`k(d) = \max(1 - d/2, 0)`.
        * Other kernels are ``'tricube'``, ``'triweight'``, ``'quartic'``,
          ``'epanechnikov'``, ``'logistic'``, and ``'sigmoid'``.
          See `Wikipedia <https://en.wikipedia.org/wiki/Kernel_(statistics)>`_.

        Another option is to pass a function that takes a vector of pairwise
        distances and returns the similarities. All the predefined kernels
        return a similarity of 0.5 when the distance is one.
        An example of custom kernel is ``kernel=lambda d: d.min() / d``.
    kernel_width : float, optional
        Control the width, also known as the bandwidth, :math:`\sigma` of the
        kernel by scaling the distances as ``distances / kernel_width`` before
        calling the kernel function.
        By default, it is set to the average of all computed distances.
        When building a radius graph, it's common to set it as a function of
        the radius, such as ``radius / 2``.
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

    kwargs : dict
        Parameters to be passed to the :class:`Graph` constructor or the
        backend library.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> features = np.random.RandomState(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(features)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    Plot all the pre-defined kernels.

    >>> width = 0.3
    >>> distances = np.linspace(0, 1, 200)
    >>> fig, ax = plt.subplots()
    >>> for name, kernel in graphs.NNGraph._kernels.items():
    ...     _ = ax.plot(distances, kernel(distances / width), label=name)
    >>> _ = ax.set_xlabel('distance [0, inf]')
    >>> _ = ax.set_ylabel('similarity [0, 1]')
    >>> _ = ax.legend(loc='upper right')

    """

    def __init__(self, features, standardize=False,
                 metric='euclidean', order=3,
                 kind='knn', k=10, radius='estimate-knn',
                 kernel='gaussian', kernel_width=None,
                 backend='scipy-ckdtree',
                 **kwargs):

        # features is stored in coords, potentially standardized
        self.standardize = standardize
        self.metric = metric
        self.kind = kind
        self.kernel = kernel
        self.k = k
        self.backend = backend

        features = np.asanyarray(features)
        if features.ndim != 2:
            raise ValueError('features should be #vertices x dimensionality')
        n_vertices, dimensionality = features.shape

        params_graph = dict()
        for key in ['lap_type', 'plotting']:
            try:
                params_graph[key] = kwargs.pop(key)
            except KeyError:
                pass

        if standardize:
            # Don't alter the original data (users would be surprised).
            features = features - np.mean(features, axis=0)
            features /= np.std(features, axis=0)

        # Order consistent with metric (used by kdtree and ckdtree).
        _orders = {
            'euclidean': 2,
            'manhattan': 1,
            'max_dist': np.inf,
            'minkowski': order,
        }
        order = _orders.pop(metric, None)

        if kind == 'knn':
            if not 1 <= k < n_vertices:
                raise ValueError('The number of neighbors (k={}) must be '
                                 'greater than 0 and smaller than the number '
                                 'of vertices ({}).'.format(k, n_vertices))
            radius = None
        elif kind == 'radius':
            if radius == 'estimate':
                maximums = np.amax(features, axis=0)
                minimums = np.amin(features, axis=0)
                distance_max = np.linalg.norm(maximums - minimums, order)
                radius = distance_max / np.power(n_vertices, 1/dimensionality)
            elif radius == 'estimate-knn':
                graph = NNGraph(features, standardize=standardize,
                                metric=metric, order=order, kind='knn', k=k,
                                kernel_width=None, backend=backend, **kwargs)
                radius = graph.kernel_width
            elif radius <= 0:
                raise ValueError('The radius must be greater than 0.')
            k = None
        else:
            raise ValueError('Invalid kind "{}".'.format(kind))

        try:
            function = globals()['_' + backend.replace('-', '_')]
        except KeyError:
            raise ValueError('Invalid backend "{}".'.format(backend))
        neighbors, distances = function(features, metric, order,
                                        kind, k, radius, kwargs)

        n_edges = [len(n) - 1 for n in neighbors]  # remove distance to self

        if kind == 'radius':
            n_disconnected = np.sum(np.asarray(n_edges) == 0)
            if n_disconnected > 0:
                logger = utils.build_logger(__name__)
                logger.warning('{} vertices (out of {}) are disconnected. '
                               'Consider increasing the radius or setting '
                               'kind=knn.'.format(n_disconnected, n_vertices))

        value = np.empty(sum(n_edges), dtype=np.float)
        row = np.empty_like(value, dtype=np.int)
        col = np.empty_like(value, dtype=np.int)
        start = 0
        for vertex in range(n_vertices):
            if kind == 'knn':
                assert n_edges[vertex] == k
            end = start + n_edges[vertex]
            value[start:end] = distances[vertex][1:]
            row[start:end] = np.full(n_edges[vertex], vertex)
            col[start:end] = neighbors[vertex][1:]
            start = end
        W = sparse.csr_matrix((value, (row, col)), (n_vertices, n_vertices))

        # Enforce symmetry. May have been broken by k-NN. Checking symmetry
        # with np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method='fill')

        if kernel_width is None:
            kernel_width = np.mean(W.data) if W.nnz > 0 else np.nan

        if not callable(kernel):
            try:
                kernel = self._kernels[kernel]
            except KeyError:
                raise ValueError('Unknown kernel {}.'.format(kernel))

        assert np.all(W.data >= 0), 'Distance must be in [0, inf].'
        W.data = kernel(W.data / kernel_width)
        if not np.all((W.data >= 0) & (W.data <= 1)):
            raise ValueError('Kernel returned similarity not in [0, 1].')

        self.order = order
        self.radius = radius
        self.kernel_width = kernel_width

        super(NNGraph, self).__init__(W=W, coords=features, **params_graph)

    def _get_extra_repr(self):
        attrs = {
            'standardize': self.standardize,
            'metric': self.metric,
            'order': self.order,
            'kind': self.kind,
        }
        if self.k is not None:
            attrs['k'] = self.k
        if self.radius is not None:
            attrs['radius'] = '{:.2e}'.format(self.radius)
        attrs.update({
            'kernel': '{}'.format(self.kernel),
            'kernel_width': '{:.2e}'.format(self.kernel_width),
            'backend': self.backend,
        })
        return attrs

    @staticmethod
    def _kernel_rectangular(distance):
        return (distance < 1).astype(np.float)

    @staticmethod
    def _kernel_triangular(distance, value_at_one=0.5):
        distance = value_at_one * distance
        return np.maximum(1 - distance, 0)

    @staticmethod
    def _kernel_exponential(distance, power=1, value_at_one=0.5):
        cst = np.log(value_at_one)
        return np.exp(cst * distance**power)

    @staticmethod
    def _kernel_powers(distance, pow1, pow2, value_at_one=0.5):
        cst = (1 - value_at_one**(1/pow2))**(1/pow1)
        distance = np.clip(cst * distance, 0, 1)
        return (1 - distance**pow1)**pow2

    @staticmethod
    def _kernel_logistic(distance, value_at_one=0.5):
        cst = 4 / value_at_one - 2
        cst = np.log(0.5 * (cst + np.sqrt(cst**2 - 4)))
        distance = cst * distance
        return 4 / (np.exp(distance) + 2 + np.exp(-distance))

    @staticmethod
    def _kernel_sigmoid(distance, value_at_one=0.5):
        cst = 2 / value_at_one
        cst = np.log(0.5 * (cst + np.sqrt(cst**2 - 4)))
        distance = cst * distance
        return 2 / (np.exp(distance) + np.exp(-distance))

    _kernels = {
        'rectangular': _kernel_rectangular.__func__,
        'triangular': _kernel_triangular.__func__,

        'exponential': _kernel_exponential.__func__,
        'gaussian': partial(_kernel_exponential.__func__, power=2),

        'tricube': partial(_kernel_powers.__func__, pow1=3, pow2=3),
        'triweight': partial(_kernel_powers.__func__, pow1=2, pow2=3),
        'quartic': partial(_kernel_powers.__func__, pow1=2, pow2=2),
        'epanechnikov': partial(_kernel_powers.__func__, pow1=2, pow2=1),

        'logistic': _kernel_logistic.__func__,
        'sigmoid': _kernel_sigmoid.__func__,
    }
