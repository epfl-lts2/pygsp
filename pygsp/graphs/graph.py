# -*- coding: utf-8 -*-

from __future__ import division

from collections import Counter

import numpy as np
from scipy import sparse

from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5


class Graph(fourier.GraphFourier, difference.GraphDifference):
    r"""Base graph class.

    * Instantiate it to construct a graph from a (weighted) adjacency matrix.
    * Provide a common interface (and implementation) for graph objects.
    * Initialize attributes for derived classes.

    Parameters
    ----------
    adjacency : sparse matrix or array_like
        The (weighted) adjacency matrix of size n_vertices by n_vertices that
        encodes the graph.
        The data is copied except if it is a sparse matrix in CSR format.
    lap_type : {'combinatorial', 'normalized'}
        The kind of Laplacian to be computed by :meth:`compute_laplacian`.
    coords : array_like
        A matrix of size n_vertices by d that represents the coordinates of the
        vertices in a d-dimensional embedding space.
    plotting : dict
        Plotting parameters.

    Attributes
    ----------
    n_vertices or N : int
        The number of vertices (nodes) in the graph.
    n_edges or Ne : int
        The number of edges (links) in the graph.
    W : :class:`scipy.sparse.csr_matrix`
        The adjacency matrix that contains the weights of the edges.
        It is represented as an n_vertices by n_vertices matrix, where
        :math:`W_{i,j}` is the weight of the edge :math:`(v_i, v_j)` from
        vertex :math:`v_i` to vertex :math:`v_j`. :math:`W_{i,j} = 0` means
        that there is no direct connection.
    L : :class:`scipy.sparse.csr_matrix`
        The graph Laplacian, an N-by-N matrix computed from W.
    lap_type : 'normalized', 'combinatorial'
        The kind of Laplacian that was computed by :func:`compute_laplacian`.
    coords : :class:`numpy.ndarray`
        Vertices coordinates in 2D or 3D space. Used for plotting only.
    plotting : dict
        Plotting parameters.

    Examples
    --------

    Define a simple graph.

    >>> graph = graphs.Graph([
    ...     [0., 2., 0.],
    ...     [2., 0., 5.],
    ...     [0., 5., 0.],
    ... ])
    >>> graph
    Graph(n_vertices=3, n_edges=2)
    >>> graph.n_vertices, graph.n_edges
    (3, 2)
    >>> graph.W.toarray()
    array([[0., 2., 0.],
           [2., 0., 5.],
           [0., 5., 0.]])
    >>> graph.d
    array([1, 2, 1], dtype=int32)
    >>> graph.dw
    array([2., 7., 5.])
    >>> graph.L.toarray()
    array([[ 2., -2.,  0.],
           [-2.,  7., -5.],
           [ 0., -5.,  5.]])

    Add some coordinates to plot it.

    >>> import matplotlib.pyplot as plt
    >>> graph.set_coordinates([
    ...     [0, 0],
    ...     [0, 1],
    ...     [1, 0],
    ... ])
    >>> fig, ax = graph.plot()

    """

    def __init__(self, adjacency, lap_type='combinatorial', coords=None,
                 plotting={}):

        self.logger = utils.build_logger(__name__)

        if not sparse.isspmatrix(adjacency):
            adjacency = np.asanyarray(adjacency)

        if (adjacency.ndim != 2) or (adjacency.shape[0] != adjacency.shape[1]):
            raise ValueError('Adjacency: must be a square matrix.')

        # CSR sparse matrices are the most efficient for matrix multiplication.
        # They are the sole sparse matrix type to support eliminate_zeros().
        self.W = sparse.csr_matrix(adjacency, copy=False)

        if np.isnan(self.W.sum()):
            raise ValueError('Adjacency: there is a Not a Number (NaN).')
        if np.isinf(self.W.sum()):
            raise ValueError('Adjacency: there is an infinite value.')
        if self.has_loops():
            self.logger.warning('Adjacency: there are self-loops '
                                '(non-zeros on the diagonal). '
                                'The Laplacian will not see them.')
        if (self.W < 0).nnz != 0:
            self.logger.warning('Adjacency: there are negative edge weights.')

        self.n_vertices = self.W.shape[0]

        # Don't keep edges of 0 weight. Otherwise n_edges will not correspond
        # to the real number of edges. Problematic when plotting.
        self.W.eliminate_zeros()

        # Don't count edges two times if undirected.
        # Be consistent with the size of the differential operator.
        if self.is_directed():
            self.n_edges = self.W.nnz
        else:
            diagonal = np.count_nonzero(self.W.diagonal())
            off_diagonal = self.W.nnz - diagonal
            self.n_edges = off_diagonal // 2 + diagonal

        self.compute_laplacian(lap_type)

        if coords is not None:
            self.coords = np.asanyarray(coords)

        self.plotting = {'vertex_size': 100,
                         'vertex_color': (0.12, 0.47, 0.71, 0.5),
                         'edge_color': (0.5, 0.5, 0.5, 0.5),
                         'edge_width': 2,
                         'edge_style': '-'}
        self.plotting.update(plotting)
        self.signals = dict()

        # TODO: kept for backward compatibility.
        self.Ne = self.n_edges
        self.N = self.n_vertices

    def _get_extra_repr(self):
        return dict()

    def __repr__(self, limit=None):
        s = ''
        for attr in ['n_vertices', 'n_edges']:
            s += '{}={}, '.format(attr, getattr(self, attr))
        for i, (key, value) in enumerate(self._get_extra_repr().items()):
            if (limit is not None) and (i == limit - 2):
                s += '..., '
                break
            s += '{}={}, '.format(key, value)
        return '{}({})'.format(self.__class__.__name__, s[:-2])


    def check_weights(self):
        r"""Check the characteristics of the weights matrix.

        Returns
        -------
        A dict of bools containing informations about the matrix

        has_inf_val : bool
            True if the matrix has infinite values else false
        has_nan_value : bool
            True if the matrix has a "not a number" value else false
        is_not_square : bool
            True if the matrix is not square else false
        diag_is_not_zero : bool
            True if the matrix diagonal has not only zeros else false

        Examples
        --------
        >>> W = np.arange(4).reshape(2, 2)
        >>> G = graphs.Graph(W)
        >>> cw = G.check_weights()
        >>> cw == {'has_inf_val': False, 'has_nan_value': False,
        ...        'is_not_square': False, 'diag_is_not_zero': True}
        True

        """

        has_inf_val = False
        diag_is_not_zero = False
        is_not_square = False
        has_nan_value = False

        if np.isinf(self.W.sum()):
            self.logger.warning('There is an infinite '
                                'value in the weight matrix!')
            has_inf_val = True

        if abs(self.W.diagonal()).sum() != 0:
            self.logger.warning('The main diagonal of '
                                'the weight matrix is not 0!')
            diag_is_not_zero = True

        if self.W.get_shape()[0] != self.W.get_shape()[1]:
            self.logger.warning('The weight matrix is not square!')
            is_not_square = True

        if np.isnan(self.W.sum()):
            self.logger.warning('There is a NaN value in the weight matrix!')
            has_nan_value = True

        return {'has_inf_val': has_inf_val,
                'has_nan_value': has_nan_value,
                'is_not_square': is_not_square,
                'diag_is_not_zero': diag_is_not_zero}

    def set_coordinates(self, kind='spring', **kwargs):
        r"""Set node's coordinates (their position when plotting).

        Parameters
        ----------
        kind : string or array_like
            Kind of coordinates to generate. It controls the position of the
            nodes when plotting the graph. Can either pass an array of size Nx2
            or Nx3 to set the coordinates manually or the name of a layout
            algorithm. Available algorithms: community2D, random2D, random3D,
            ring2D, line1D, spring, laplacian_eigenmap2D, laplacian_eigenmap3D.
            Default is 'spring'.
        kwargs : dict
            Additional parameters to be passed to the Fruchterman-Reingold
            force-directed algorithm when kind is spring.

        Examples
        --------
        >>> G = graphs.ErdosRenyi()
        >>> G.set_coordinates()
        >>> fig, ax = G.plot()

        """

        if not isinstance(kind, str):
            coords = np.asanyarray(kind).squeeze()
            check_1d = (coords.ndim == 1)
            check_2d_3d = (coords.ndim == 2) and (2 <= coords.shape[1] <= 3)
            if coords.shape[0] != self.N or not (check_1d or check_2d_3d):
                raise ValueError('Expecting coordinates to be of size N, Nx2, '
                                 'or Nx3.')
            self.coords = coords

        elif kind == 'line1D':
            self.coords = np.arange(self.N)

        elif kind == 'line2D':
            x, y = np.arange(self.N), np.zeros(self.N)
            self.coords = np.stack([x, y], axis=1)

        elif kind == 'ring2D':
            angle = np.arange(self.N) * 2 * np.pi / self.N
            self.coords = np.stack([np.cos(angle), np.sin(angle)], axis=1)

        elif kind == 'random2D':
            self.coords = np.random.uniform(size=(self.N, 2))

        elif kind == 'random3D':
            self.coords = np.random.uniform(size=(self.N, 3))

        elif kind == 'spring':
            self.coords = self._fruchterman_reingold_layout(**kwargs)

        elif kind == 'community2D':
            if not hasattr(self, 'info') or 'node_com' not in self.info:
                ValueError('Missing arguments to the graph to be able to '
                           'compute community coordinates.')

            if 'world_rad' not in self.info:
                self.info['world_rad'] = np.sqrt(self.N)

            if 'comm_sizes' not in self.info:
                counts = Counter(self.info['node_com'])
                self.info['comm_sizes'] = np.array([cnt[1] for cnt
                                                    in sorted(counts.items())])

            Nc = self.info['comm_sizes'].shape[0]

            self.info['com_coords'] = self.info['world_rad'] * \
                np.array(list(zip(
                    np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc),
                    np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))

            # Coordinates of the nodes inside their communities
            coords = np.random.rand(self.N, 2)
            self.coords = np.array([[elem[0] * np.cos(2 * np.pi * elem[1]),
                                     elem[0] * np.sin(2 * np.pi * elem[1])]
                                    for elem in coords])

            for i in range(self.N):
                # Set coordinates as an offset from the center of the community
                # it belongs to
                comm_idx = self.info['node_com'][i]
                comm_rad = np.sqrt(self.info['comm_sizes'][comm_idx])
                self.coords[i] = self.info['com_coords'][comm_idx] + \
                    comm_rad * self.coords[i]
        elif kind == 'laplacian_eigenmap2D':
            self.compute_fourier_basis(n_eigenvectors=2)
            self.coords = self.U[:, 1:3]
        elif kind == 'laplacian_eigenmap3D':
            self.compute_fourier_basis(n_eigenvectors=3)
            self.coords = self.U[:, 1:4]
        else:
            raise ValueError('Unexpected argument kind={}.'.format(kind))

    def subgraph(self, vertices):
        r"""Create a subgraph from a list of vertices.

        Parameters
        ----------
        vertices : list
            List of vertices to keep.

        Returns
        -------
        subgraph : :class:`Graph`
            Subgraph.

        Examples
        --------
        >>> graph = graphs.Graph([
        ...     [0., 3., 0., 0.],
        ...     [3., 0., 4., 0.],
        ...     [0., 4., 0., 2.],
        ...     [0., 0., 2., 0.],
        ... ])
        >>> graph = graph.subgraph([0, 2, 1])
        >>> graph.W.toarray()
        array([[0., 0., 3.],
               [0., 0., 4.],
               [3., 4., 0.]])

        """
        adjacency = self.W[vertices, :][:, vertices]
        try:
            coords = self.coords[vertices]
        except AttributeError:
            coords = None
        return Graph(adjacency, self.lap_type, coords, self.plotting)

    def is_connected(self, recompute=False):
        r"""Check if the graph is connected (cached).

        A graph is connected if and only if there exists a (directed) path
        between any two vertices.

        Parameters
        ----------
        recompute: bool
            Force to recompute the connectivity if already known.

        Returns
        -------
        connected : bool
            True if the graph is connected, False otherwise.

        Notes
        -----

        For undirected graphs, starting at a vertex and trying to visit all the
        others is enough.
        For directed graphs, one needs to check that a vertex can both be
        visited by all the others and visit all the others.

        Examples
        --------

        Connected graph:

        >>> graph = graphs.Graph([
        ...     [0, 3, 0, 0],
        ...     [3, 0, 4, 0],
        ...     [0, 4, 0, 2],
        ...     [0, 0, 2, 0],
        ... ])
        >>> graph.is_connected()
        True

        Disconnected graph:

        >>> graph = graphs.Graph([
        ...     [0, 3, 0, 0],
        ...     [3, 0, 4, 0],
        ...     [0, 0, 0, 2],
        ...     [0, 0, 2, 0],
        ... ])
        >>> graph.is_connected()
        False


        """
        if hasattr(self, '_connected') and not recompute:
            return self._connected

        adjacencies = [self.W]
        if self.is_directed(recompute=recompute):
            adjacencies.append(self.W.T)

        for adjacency in adjacencies:
            visited = np.zeros(self.n_vertices, dtype=np.bool)
            stack = set([0])

            while stack:
                vertex = stack.pop()

                if visited[vertex]:
                    continue
                visited[vertex] = True

                neighbors = adjacency[vertex].nonzero()[1]
                stack.update(neighbors)

            if not np.all(visited):
                self._connected = False
                return self._connected

        self._connected = True
        return self._connected

    def is_directed(self, recompute=False):
        r"""Check if the graph has directed edges (cached).

        In this framework, we consider that a graph is directed if and
        only if its weight matrix is not symmetric.

        Parameters
        ----------
        recompute : bool
            Force to recompute the directedness if already known.

        Returns
        -------
        directed : bool
            True if the graph is directed, False otherwise.

        Examples
        --------

        Directed graph:

        >>> graph = graphs.Graph([
        ...     [0, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 0, 0],
        ... ])
        >>> graph.is_directed()
        True

        Undirected graph:

        >>> graph = graphs.Graph([
        ...     [0, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 4, 0],
        ... ])
        >>> graph.is_directed()
        False

        """
        if hasattr(self, '_directed') and not recompute:
            return self._directed

        self._directed = (self.W != self.W.T).nnz != 0
        return self._directed

    def has_loops(self):
        r"""Check if any vertex is connected to itself.

        A graph has self-loops if and only if the diagonal entries of its
        adjacency matrix are not all zero.

        Returns
        -------
        loops : bool
            True if the graph has self-loops, False otherwise.

        Examples
        --------

        Without self-loops:

        >>> graph = graphs.Graph([
        ...     [0, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 0, 0],
        ... ])
        >>> graph.has_loops()
        False

        With a self-loop:

        >>> graph = graphs.Graph([
        ...     [1, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 0, 0],
        ... ])
        >>> graph.has_loops()
        True

        """
        return np.any(self.W.diagonal() != 0)

    def extract_components(self):
        r"""Split the graph into connected components.

        See :func:`is_connected` for the method used to determine
        connectedness.

        Returns
        -------
        graphs : list
            A list of graph structures. Each having its own node list and
            weight matrix. If the graph is directed, add into the info
            parameter the information about the source nodes and the sink
            nodes.

        Examples
        --------
        >>> from scipy import sparse
        >>> W = sparse.rand(10, 10, 0.2)
        >>> W = utils.symmetrize(W)
        >>> G = graphs.Graph(W)
        >>> components = G.extract_components()
        >>> has_sinks = 'sink' in components[0].info
        >>> sinks_0 = components[0].info['sink'] if has_sinks else []

        """
        if self.A.shape[0] != self.A.shape[1]:
            self.logger.error('Inconsistent shape to extract components. '
                              'Square matrix required.')
            return None

        if self.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(self.A.shape[0], dtype=bool)
        # indices = [] # Assigned but never used

        while not visited.all():
            # pick a node not visted yet
            stack = set(np.nonzero(~visited)[0][[0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    # Add indices of nodes not visited yet and accessible from
                    # v
                    stack.update(set([idx for idx in self.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            self.logger.info(('Constructing subgraph for component of '
                              'size {}.').format(len(comp)))
            G = self.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

    def compute_laplacian(self, lap_type='combinatorial'):
        r"""Compute a graph Laplacian.

        For undirected graphs, the combinatorial Laplacian is defined as

        .. math:: L = D - W,

        where :math:`W` is the weighted adjacency matrix and :math:`D` the
        weighted degree matrix. The normalized Laplacian is defined as

        .. math:: L = I - D^{-1/2} W D^{-1/2},

        where :math:`I` is the identity matrix.

        For directed graphs, the Laplacians are built from a symmetrized
        version of the weighted adjacency matrix that is the average of the
        weighted adjacency matrix and its transpose. As the Laplacian is
        defined as the divergence of the gradient, it is not affected by the
        orientation of the edges.

        For both Laplacians, the diagonal entries corresponding to disconnected
        nodes (i.e., nodes with degree zero) are set to zero.

        Once computed, the Laplacian is accessible by the attribute :attr:`L`.

        Parameters
        ----------
        lap_type : {'combinatorial', 'normalized'}
            The kind of Laplacian to compute. Default is combinatorial.

        Examples
        --------

        Combinatorial and normalized Laplacians of an undirected graph.

        >>> graph = graphs.Graph([
        ...     [0, 2, 0],
        ...     [2, 0, 1],
        ...     [0, 1, 0],
        ... ])
        >>> graph.compute_laplacian('combinatorial')
        >>> graph.L.toarray()
        array([[ 2., -2.,  0.],
               [-2.,  3., -1.],
               [ 0., -1.,  1.]])
        >>> graph.compute_laplacian('normalized')
        >>> graph.L.toarray()
        array([[ 1.        , -0.81649658,  0.        ],
               [-0.81649658,  1.        , -0.57735027],
               [ 0.        , -0.57735027,  1.        ]])

        Combinatorial and normalized Laplacians of a directed graph.

        >>> graph = graphs.Graph([
        ...     [0, 2, 0],
        ...     [2, 0, 1],
        ...     [0, 0, 0],
        ... ])
        >>> graph.compute_laplacian('combinatorial')
        >>> graph.L.toarray()
        array([[ 2. , -2. ,  0. ],
               [-2. ,  2.5, -0.5],
               [ 0. , -0.5,  0.5]])
        >>> graph.compute_laplacian('normalized')
        >>> graph.L.toarray()
        array([[ 1.        , -0.89442719,  0.        ],
               [-0.89442719,  1.        , -0.4472136 ],
               [ 0.        , -0.4472136 ,  1.        ]])

        The Laplacian is defined as the divergence of the gradient.
        See :meth:`compute_differential_operator` for details.

        >>> graph = graphs.Path(20)
        >>> graph.compute_differential_operator()
        >>> L = graph.D.dot(graph.D.T)
        >>> np.all(L.toarray() == graph.L.toarray())
        True

        The Laplacians have a bounded spectrum.

        >>> G = graphs.Sensor(50)
        >>> G.compute_laplacian('combinatorial')
        >>> G.compute_fourier_basis()
        >>> -1e-10 < G.e[0] < 1e-10 < G.e[-1] < 2*np.max(G.dw)
        True
        >>> G.compute_laplacian('normalized')
        >>> G.compute_fourier_basis(recompute=True)
        >>> -1e-10 < G.e[0] < 1e-10 < G.e[-1] < 2
        True

        """

        self.lap_type = lap_type

        if not self.is_directed():
            W = self.W
        else:
            W = utils.symmetrize(self.W, method='average')

        if lap_type == 'combinatorial':
            D = sparse.diags(self.dw)
            self.L = D - W
        elif lap_type == 'normalized':
            d = np.zeros(self.n_vertices)
            disconnected = (self.dw == 0)
            np.power(self.dw, -0.5, where=~disconnected, out=d)
            D = sparse.diags(d)
            self.L = sparse.identity(self.n_vertices) - D * W * D
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        else:
            raise ValueError('Unknown Laplacian type {}'.format(lap_type))

    def _check_signal(self, s):
        r"""Check if signal is valid."""
        s = np.asanyarray(s)
        if s.shape[0] != self.N:
            raise ValueError('First dimension must be the number of vertices '
                             'G.N = {}, got {}.'.format(self.N, s.shape))
        return s

    def dirichlet_energy(self, x):
        r"""Compute the Dirichlet energy of a signal defined on the vertices.

        The Dirichlet energy of a signal :math:`x` is defined as

        .. math:: x^\top L x = \| \nabla_\mathcal{G} x \|_2^2
                             = \frac12 \sum_{i,j} W[i, j] (x[j] - x[i])^2

        for the combinatorial Laplacian, and

        .. math:: x^\top L x = \| \nabla_\mathcal{G} x \|_2^2
            = \frac12 \sum_{i,j} W[i, j]
              \left( \frac{x[j]}{d[j]} - \frac{x[i]}{d[i]} \right)^2

        for the normalized Laplacian, where :math:`d` is the weighted degree
        :attr:`dw`, :math:`\nabla_\mathcal{G} x = D^\top x` and :math:`D` is
        the differential operator :attr:`D`. See :meth:`grad` for the
        definition of the gradient :math:`\nabla_\mathcal{G}`.

        Parameters
        ----------
        x : array_like
            Signal of length :attr:`n_vertices` living on the vertices.

        Returns
        -------
        energy : float
            The Dirichlet energy of the graph signal.

        See also
        --------
        grad : compute the gradient of a vertex signal

        Examples
        --------
        >>> graph = graphs.Path(5, directed=False)
        >>> signal = [0, 2, 2, 4, 4]
        >>> graph.dirichlet_energy(signal)
        8.0
        >>> # The Dirichlet energy is indeed the squared norm of the gradient.
        >>> graph.compute_differential_operator()
        >>> graph.grad(signal)
        array([2., 0., 2., 0.])

        >>> graph = graphs.Path(5, directed=True)
        >>> signal = [0, 2, 2, 4, 4]
        >>> graph.dirichlet_energy(signal)
        4.0
        >>> # The Dirichlet energy is indeed the squared norm of the gradient.
        >>> graph.compute_differential_operator()
        >>> graph.grad(signal)
        array([1.41421356, 0.        , 1.41421356, 0.        ])

        """
        x = self._check_signal(x)
        return x.T.dot(self.L.dot(x))

    @property
    def A(self):
        r"""Graph adjacency matrix (the binary version of W).

        The adjacency matrix defines which edges exist on the graph.
        It is represented as an N-by-N matrix of booleans.
        :math:`A_{i,j}` is True if :math:`W_{i,j} > 0`.
        """
        if not hasattr(self, '_A'):
            self._A = self.W > 0
        return self._A

    @property
    def d(self):
        r"""The degree (number of neighbors) of vertices.

        For undirected graphs, the degree of a vertex is the number of vertices
        it is connected to.
        For directed graphs, the degree is the average of the in and out
        degrees, where the in degree is the number of incoming edges, and the
        out degree the number of outgoing edges.

        In both cases, the degree of the vertex :math:`v_i` is the average
        between the number of non-zero values in the :math:`i`-th column (the
        in degree) and the :math:`i`-th row (the out degree) of the weighted
        adjacency matrix :attr:`W`.

        Examples
        --------

        Undirected graph:

        >>> graph = graphs.Graph([
        ...     [0, 1, 0],
        ...     [1, 0, 2],
        ...     [0, 2, 0],
        ... ])
        >>> print(graph.d)  # Number of neighbors.
        [1 2 1]
        >>> print(graph.dw)  # Weighted degree.
        [1 3 2]

        Directed graph:

        >>> graph = graphs.Graph([
        ...     [0, 1, 0],
        ...     [0, 0, 2],
        ...     [0, 2, 0],
        ... ])
        >>> print(graph.d)  # Number of neighbors.
        [0.5 1.5 1. ]
        >>> print(graph.dw)  # Weighted degree.
        [0.5 2.5 2. ]

        """
        if not hasattr(self, '_d'):
            if not self.is_directed():
                # Shortcut for undirected graphs.
                self._d = self.W.getnnz(axis=1)
                # axis=1 faster for CSR (https://stackoverflow.com/a/16391764)
            else:
                degree_in = self.W.getnnz(axis=0)
                degree_out = self.W.getnnz(axis=1)
                self._d = (degree_in + degree_out) / 2
        return self._d

    @property
    def dw(self):
        r"""The weighted degree of vertices.

        For undirected graphs, the weighted degree of the vertex :math:`v_i` is
        defined as

        .. math:: d[i] = \sum_j W[j, i] = \sum_j W[i, j],

        where :math:`W` is the weighted adjacency matrix :attr:`W`.

        For directed graphs, the weighted degree of the vertex :math:`v_i` is
        defined as

        .. math:: d[i] = \frac12 (d^\text{in}[i] + d^\text{out}[i])
                       = \frac12 (\sum_j W[j, i] + \sum_j W[i, j]),

        i.e., as the average of the in and out degrees.

        Examples
        --------

        Undirected graph:

        >>> graph = graphs.Graph([
        ...     [0, 1, 0],
        ...     [1, 0, 2],
        ...     [0, 2, 0],
        ... ])
        >>> print(graph.d)  # Number of neighbors.
        [1 2 1]
        >>> print(graph.dw)  # Weighted degree.
        [1 3 2]

        Directed graph:

        >>> graph = graphs.Graph([
        ...     [0, 1, 0],
        ...     [0, 0, 2],
        ...     [0, 2, 0],
        ... ])
        >>> print(graph.d)  # Number of neighbors.
        [0.5 1.5 1. ]
        >>> print(graph.dw)  # Weighted degree.
        [0.5 2.5 2. ]

        """
        if not hasattr(self, '_dw'):
            if not self.is_directed():
                # Shortcut for undirected graphs.
                self._dw = np.ravel(self.W.sum(axis=0))
            else:
                degree_in = np.ravel(self.W.sum(axis=0))
                degree_out = np.ravel(self.W.sum(axis=1))
                self._dw = (degree_in + degree_out) / 2
        return self._dw

    @property
    def lmax(self):
        r"""Largest eigenvalue of the graph Laplacian.

        Can be exactly computed by :func:`compute_fourier_basis` or
        approximated by :func:`estimate_lmax`.
        """
        if not hasattr(self, '_lmax'):
            self.logger.warning('The largest eigenvalue G.lmax is not '
                                'available, we need to estimate it. '
                                'Explicitly call G.estimate_lmax() or '
                                'G.compute_fourier_basis() '
                                'once beforehand to suppress the warning.')
            self.estimate_lmax()
        return self._lmax

    def estimate_lmax(self, method='lanczos', recompute=False):
        r"""Estimate the Laplacian's largest eigenvalue (cached).

        The result is cached and accessible by the :attr:`lmax` property.

        Exact value given by the eigendecomposition of the Laplacian, see
        :func:`compute_fourier_basis`. That estimation is much faster than the
        eigendecomposition.

        Parameters
        ----------
        method : {'lanczos', 'bounds'}
            Whether to estimate the largest eigenvalue with the implicitly
            restarted Lanczos method, or to return an upper bound on the
            spectrum of the Laplacian.
        recompute : boolean
            Force to recompute the largest eigenvalue. Default is false.

        Notes
        -----
        Runs the implicitly restarted Lanczos method (as implemented in
        :func:`scipy.sparse.linalg.eigsh`) with a large tolerance, then
        increases the calculated largest eigenvalue by 1 percent. For much of
        the PyGSP machinery, we need to approximate filter kernels on an
        interval that contains the spectrum of L. The only cost of using a
        larger interval is that the polynomial approximation over the larger
        interval may be a slightly worse approximation on the actual spectrum.
        As this is a very mild effect, it is not necessary to obtain very tight
        bounds on the spectrum of L.

        A faster but less tight alternative is to use known algebraic bounds on
        the graph Laplacian.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()  # True value.
        >>> print('{:.2f}'.format(G.lmax))
        13.78
        >>> G.estimate_lmax(recompute=True)  # Estimate.
        >>> print('{:.2f}'.format(G.lmax))
        13.92
        >>> G.estimate_lmax(method='bounds', recompute=True)  # Upper bound.
        >>> print('{:.2f}'.format(G.lmax))
        18.58

        """
        if hasattr(self, '_lmax') and not recompute:
            return

        if method == 'lanczos':
            try:
                # We need to cast the matrix L to a supported type.
                # TODO: not good for memory. Cast earlier?
                lmax = sparse.linalg.eigsh(self.L.asfptype(), k=1, tol=5e-3,
                                           ncv=min(self.N, 10),
                                           return_eigenvectors=False)
                lmax = lmax[0]
                assert lmax <= self._get_upper_bound() + 1e-12
                lmax *= 1.01  # Increase by 1% to be robust to errors.
                self._lmax = lmax
            except sparse.linalg.ArpackNoConvergence:
                raise ValueError('The Lanczos method did not converge. '
                                 'Try to use bounds.')

        elif method == 'bounds':
            self._lmax = self._get_upper_bound()

        else:
            raise ValueError('Unknown method {}'.format(method))

    def _get_upper_bound(self):
        r"""Return an upper bound on the eigenvalues of the Laplacian."""

        if self.lap_type == 'normalized':
            return 2  # Equal iff the graph is bipartite.
        elif self.lap_type == 'combinatorial':
            bounds = []
            # Equal for full graphs.
            bounds += [self.n_vertices * np.max(self.W)]
            # Gershgorin circle theorem. Equal for regular bipartite graphs.
            # Special case of the below bound.
            bounds += [2 * np.max(self.dw)]
            # Anderson, Morley, Eigenvalues of the Laplacian of a graph.
            # Equal for regular bipartite graphs.
            if self.n_edges > 0:
                sources, targets, _ = self.get_edge_list()
                bounds += [np.max(self.dw[sources] + self.dw[targets])]
            # Merris, A note on Laplacian graph eigenvalues.
            if not self.is_directed():
                W = self.W
            else:
                W = utils.symmetrize(self.W, method='average')
            m = W.dot(self.dw) / self.dw  # Mean degree of adjacent vertices.
            bounds += [np.max(self.dw + m)]
            # Good review: On upper bounds for Laplacian graph eigenvalues.
            return min(bounds)
        else:
            raise ValueError('Unknown Laplacian type '
                             '{}'.format(self.lap_type))

    def get_edge_list(self):
        r"""Return an edge list, an alternative representation of the graph.

        Each edge :math:`e_k = (v_i, v_j) \in \mathcal{E}` from :math:`v_i` to
        :math:`v_j` is associated with the weight :math:`W[i, j]`. For each
        edge :math:`e_k`, the method returns :math:`(i, j, W[i, j])` as
        `(sources[k], targets[k], weights[k])`, with :math:`i \in [0,
        |\mathcal{V}|-1], j \in [0, |\mathcal{V}|-1], k \in [0,
        |\mathcal{E}|-1]`.

        Returns
        -------
        sources : vector of int
            Source node indices.
        targets : vector of int
            Target node indices.
        weights : vector of float
            Edge weights.

        Notes
        -----
        The weighted adjacency matrix is the canonical form used in this
        package to represent a graph as it is the easiest to work with when
        considering spectral methods.

        Edge orientation (i.e., which node is the source or the target) is
        arbitrary for undirected graphs.
        The implementation uses the upper triangular part of the adjacency
        matrix, hence :math:`i \leq j \ \forall k`.

        Examples
        --------

        Edge list of a directed graph.

        >>> graph = graphs.Graph([
        ...     [0, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 0, 0],
        ... ])
        >>> sources, targets, weights = graph.get_edge_list()
        >>> list(sources), list(targets), list(weights)
        ([0, 1, 1], [1, 0, 2], [3, 3, 4])

        Edge list of an undirected graph.

        >>> graph = graphs.Graph([
        ...     [0, 3, 0],
        ...     [3, 0, 4],
        ...     [0, 4, 0],
        ... ])
        >>> sources, targets, weights = graph.get_edge_list()
        >>> list(sources), list(targets), list(weights)
        ([0, 1], [1, 2], [3, 4])

        """

        if self.is_directed():
            W = self.W.tocoo()
        else:
            W = sparse.triu(self.W, format='coo')

        sources = W.row
        targets = W.col
        weights = W.data

        assert self.n_edges == sources.size == targets.size == weights.size
        return sources, targets, weights

    def plot(self, vertex_color=None, vertex_size=None, highlight=[],
             edges=None, edge_color=None, edge_width=None,
             indices=False, colorbar=True, limits=None, ax=None,
             title=None, backend=None):
        r"""Docstring overloaded at import time."""
        from pygsp.plotting import _plot_graph
        return _plot_graph(self, vertex_color=vertex_color,
                           vertex_size=vertex_size, highlight=highlight,
                           edges=edges, indices=indices, colorbar=colorbar,
                           edge_color=edge_color, edge_width=edge_width,
                           limits=limits, ax=ax, title=title, backend=backend)

    def plot_signal(self, *args, **kwargs):
        r"""Deprecated, use plot() instead."""
        return self.plot(*args, **kwargs)

    def plot_spectrogram(self, node_idx=None):
        r"""Docstring overloaded at import time."""
        from pygsp.plotting import _plot_spectrogram
        _plot_spectrogram(self, node_idx=node_idx)

    def _fruchterman_reingold_layout(self, dim=2, k=None, pos=None, fixed=[],
                                     iterations=50, scale=1.0, center=None,
                                     seed=None):
        # TODO doc
        # fixed: list of nodes with fixed coordinates
        # Position nodes using Fruchterman-Reingold force-directed algorithm.

        if center is None:
            center = np.zeros((1, dim))

        if np.shape(center)[1] != dim:
            self.logger.error('Spring coordinates: center has wrong size.')
            center = np.zeros((1, dim))

        if pos is None:
            dom_size = 1
            pos_arr = None
        else:
            # Determine size of existing domain to adjust initial positions
            dom_size = np.max(pos)
            pos_arr = np.random.RandomState(seed).uniform(size=(self.N, dim))
            pos_arr = pos_arr * dom_size + center
            for i in range(self.N):
                pos_arr[i] = np.asanyarray(pos[i])

        if k is None and len(fixed) > 0:
            # We must adjust k by domain size for layouts that are not near 1x1
            k = dom_size / np.sqrt(self.N)

        pos = _sparse_fruchterman_reingold(self.A, dim, k, pos_arr,
                                           fixed, iterations, seed)

        if len(fixed) == 0:
            pos = _rescale_layout(pos, scale=scale) + center

        return pos


def _sparse_fruchterman_reingold(A, dim, k, pos, fixed, iterations, seed):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    nnodes = A.shape[0]

    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except Exception:
        A = (sparse.coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.random.RandomState(seed).uniform(size=(nnodes, dim))

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)

    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    t = 0.1
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(nnodes):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = A[i, :].toarray()
            # displacement "force"
            displacement[:, i] += \
                (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        pos += (displacement * t / length).T
        # cool temperature
        t -= dt

    return pos


def _rescale_layout(pos, scale=1):
    # rescale to (-scale, scale) in all axes

    # shift origin to (0,0)
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(pos[:, i].max(), lim)
    # rescale to (-scale,scale) in all directions, preserves aspect
    for i in range(pos.shape[1]):
        pos[:, i] *= scale / lim
    return pos
