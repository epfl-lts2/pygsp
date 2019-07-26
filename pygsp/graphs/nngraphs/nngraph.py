# -*- coding: utf-8 -*-

from __future__ import division

from functools import partial

import numpy as np
from scipy import sparse, spatial

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5
from pygsp._nearest_neighbor import nearest_neighbor, sparse_distance_matrix


_logger = utils.build_logger(__name__)


class NNGraph(Graph):
    r"""Nearest-neighbor graph.

    The nearest-neighbor graph is built from a set of features, where the edge
    weight between vertices :math:`v_i` and :math:`v_j` is given by

    .. math:: A(i,j) = k \left( \frac{d(v_i, v_j)}{\sigma} \right),

    where :math:`d(v_i, v_j)` is a distance measure between some representation
    (the features) of :math:`v_i` and :math:`v_j`, :math:`k` is a kernel
    function that transforms a distance in :math:`[0, \infty]` to a similarity
    measure generally in :math:`[0, 1]`, and :math:`\sigma` is the kernel width.

    For example, the features might be the 3D coordinates of points in a point
    cloud. Then, if ``metric='euclidean'`` and ``kernel='gaussian'`` (the
    defaults), :math:`A(i,j) = \exp(-\log(2) \| x_i - x_j \|_2^2 / \sigma^2)`,
    where :math:`x_i` is the 3D position of vertex :math:`v_i`.

    The similarity matrix :math:`A` is sparsified by either keeping the ``k``
    closest vertices for each vertex (if ``type='knn'``), or by setting to zero
    the similarity when the distance is greater than ``radius`` (if ``type='radius'``).

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
        kernel. It scales the distances as ``distances / kernel_width`` before
        calling the kernel function.
        By default, it is set to the average of all computed distances for
        ``kind='knn'`` and to half the radius for ``kind='radius'``.
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
    kwargs : dict
        Parameters to be passed to the :class:`Graph` constructor or the
        backend library.

    Examples
    --------

    Construction of a graph from a set of features.

    >>> import matplotlib.pyplot as plt
    >>> rs = np.random.RandomState(42)
    >>> features = rs.uniform(size=(30, 2))
    >>> G = graphs.NNGraph(features)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    Radius versus knn graph.

    >>> features = rs.uniform(size=(100, 3))
    >>> fig, ax = plt.subplots()
    >>> G = graphs.NNGraph(features, kind='radius', radius=0.2964)
    >>> label = 'radius graph ({} edges)'.format(G.n_edges)
    >>> _ = ax.hist(G.W.data, bins=20, label=label, alpha=0.5)
    >>> G = graphs.NNGraph(features, kind='knn', k=6)
    >>> label = 'knn graph ({} edges)'.format(G.n_edges)
    >>> _ = ax.hist(G.W.data, bins=20, label=label, alpha=0.5)
    >>> _ = ax.legend()
    >>> _ = ax.set_title('edge weights')

    Control of the sparsity of knn and radius graphs.

    >>> features = rs.uniform(size=(100, 3))
    >>> n_edges = dict(knn=[], radius=[])
    >>> n_neighbors = np.arange(1, 100, 5)
    >>> radiuses = np.arange(0.05, 1.5, 0.05)
    >>> for k in n_neighbors:
    ...     G = graphs.NNGraph(features, kind='knn', k=k)
    ...     n_edges['knn'].append(G.n_edges)
    >>> for radius in radiuses:
    ...     G = graphs.NNGraph(features, kind='radius', radius=radius)
    ...     n_edges['radius'].append(G.n_edges)
    >>> fig, axes = plt.subplots(1, 2, sharey=True)
    >>> _ = axes[0].plot(n_neighbors, n_edges['knn'])
    >>> _ = axes[1].plot(radiuses, n_edges['radius'])
    >>> _ = axes[0].set_ylabel('number of edges')
    >>> _ = axes[0].set_xlabel('number of neighbors (knn graph)')
    >>> _ = axes[1].set_xlabel('radius (radius graph)')
    >>> _ = fig.suptitle('Sparsity')

    Choice of metric and the curse of dimensionality.

    >>> fig, axes = plt.subplots(1, 2)
    >>> for dim, ax in zip([3, 30], axes):
    ...     features = rs.uniform(size=(100, dim))
    ...     for metric in ['euclidean', 'manhattan', 'max_dist', 'cosine']:
    ...         G = graphs.NNGraph(features, metric=metric,
    ...                            backend='scipy-pdist')
    ...         _ = ax.hist(G.W.data, bins=20, label=metric, alpha=0.5)
    ...     _ = ax.legend()
    ...     _ = ax.set_title('edge weights, {} dimensions'.format(dim))

    Choice of kernel.

    >>> fig, axes = plt.subplots(1, 2)
    >>> width = 0.3
    >>> distances = np.linspace(0, 1, 200)
    >>> for name, kernel in graphs.NNGraph._kernels.items():
    ...     _ = axes[0].plot(distances, kernel(distances / width), label=name)
    >>> _ = axes[0].set_xlabel('distance [0, inf]')
    >>> _ = axes[0].set_ylabel('similarity [0, 1]')
    >>> _ = axes[0].legend(loc='upper right')
    >>> features = rs.uniform(size=(100, 3))
    >>> for kernel in ['gaussian', 'triangular', 'tricube', 'exponential']:
    ...     G = graphs.NNGraph(features, kernel=kernel)
    ...     _ = axes[1].hist(G.W.data, bins=20, label=kernel, alpha=0.5)
    >>> _ = axes[1].legend()
    >>> _ = axes[1].set_title('edge weights')

    Choice of kernel width.

    >>> fig, axes = plt.subplots()
    >>> for width in [.2, .3, .4, .6, .8, None]:
    ...     G = graphs.NNGraph(features, kernel_width=width)
    ...     label = 'width = {:.2f}'.format(G.kernel_width)
    ...     _ = axes.hist(G.W.data, bins=20, label=label, alpha=0.5)
    >>> _ = axes.legend(loc='upper left')
    >>> _ = axes.set_title('edge weights')

    Choice of backend. Compare on your data!

    >>> import time
    >>> sizes = [300, 1000, 3000]
    >>> dims = [3, 100]
    >>> backends = ['scipy-pdist', 'scipy-kdtree', 'scipy-ckdtree', 'flann',
    ...             'nmslib']
    >>> times = np.full((len(sizes), len(dims), len(backends)), np.nan)
    >>> for i, size in enumerate(sizes):
    ...     for j, dim in enumerate(dims):
    ...         for k, backend in enumerate(backends):
    ...             if (size * dim) > 1e4 and backend == 'scipy-kdtree':
    ...                 continue  # too slow
    ...             features = rs.uniform(size=(size, dim))
    ...             start = time.time()
    ...             _ = graphs.NNGraph(features, backend=backend)
    ...             times[i][j][k] = time.time() - start
    >>> fig, axes = plt.subplots(1, 2, sharey=True)
    >>> for j, (dim, ax) in enumerate(zip(dims, axes)):
    ...     for k, backend in enumerate(backends):
    ...         _ = ax.loglog(sizes, times[:, j, k], '.-', label=backend)
    ...         _ = ax.set_title('{} dimensions'.format(dim))
    ...         _ = ax.set_xlabel('number of vertices')
    >>> _ = axes[0].set_ylabel('execution time [s]')
    >>> _ = axes[1].legend(loc='upper left')

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
            self.k = None
        else:
            raise ValueError('Invalid kind "{}".'.format(kind))

        neighbors, distances = nearest_neighbor(features, metric=metric, order=order,
                                        kind=kind, k=k, radius=radius, backend=backend, **kwargs)
        W = sparse_distance_matrix(neighbors, distances, symmetrize=True, safe=True, kind = kind, k=k)

        if kernel_width is None:
            if kind == 'knn':
                kernel_width = np.mean(W.data) if W.nnz > 0 else np.nan
            elif kind == 'radius':
                kernel_width = radius / 2

        if not callable(kernel):
            try:
                kernel = self._kernels[kernel]
            except KeyError:
                raise ValueError('Unknown kernel {}.'.format(kernel))

        assert np.all(W.data >= 0), 'Distance must be in [0, inf].'
        W.data = kernel(W.data / kernel_width)
        if not np.all((W.data >= 0) & (W.data <= 1)):
            _logger.warning('Kernel returned similarity not in [0, 1].')

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
