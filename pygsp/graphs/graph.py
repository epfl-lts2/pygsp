import os
from collections import Counter

import numpy as np
from scipy import sparse

from pygsp import utils

from ._io import IOMixIn
from ._layout import LayoutMixIn
from .difference import DifferenceMixIn
from .fourier import FourierMixIn


class Graph(FourierMixIn, DifferenceMixIn, IOMixIn, LayoutMixIn):
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
    signals : dict (string -> :class:`numpy.ndarray`)
        Signals attached to the graph.
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

    def __init__(self, adjacency, lap_type="combinatorial", coords=None, plotting={}):
        self.logger = utils.build_logger(__name__)

        if not sparse.issparse(adjacency):
            adjacency = np.asanyarray(adjacency)

        if (adjacency.ndim != 2) or (adjacency.shape[0] != adjacency.shape[1]):
            raise ValueError("Adjacency: must be a square matrix.")

        # CSR sparse matrices are the most efficient for matrix multiplication.
        # They are the sole sparse matrix type to support eliminate_zeros().
        self._adjacency = sparse.csr_matrix(adjacency, copy=False)

        if np.isnan(self._adjacency.sum()):
            raise ValueError("Adjacency: there is a Not a Number (NaN).")
        if np.isinf(self._adjacency.sum()):
            raise ValueError("Adjacency: there is an infinite value.")
        if self.has_loops():
            self.logger.warning(
                "Adjacency: there are self-loops "
                "(non-zeros on the diagonal). "
                "The Laplacian will not see them."
            )
        if (self._adjacency < 0).nnz != 0:
            self.logger.warning("Adjacency: there are negative edge weights.")

        self.n_vertices = self._adjacency.shape[0]

        # Don't keep edges of 0 weight. Otherwise n_edges will not correspond
        # to the real number of edges. Problematic when plotting.
        self._adjacency.eliminate_zeros()

        self._directed = None
        self._connected = None

        # Don't count edges two times if undirected.
        # Be consistent with the size of the differential operator.
        if self.is_directed():
            self.n_edges = self._adjacency.nnz
        else:
            diagonal = np.count_nonzero(self._adjacency.diagonal())
            off_diagonal = self._adjacency.nnz - diagonal
            self.n_edges = off_diagonal // 2 + diagonal

        if coords is not None:
            # TODO: self.coords should be None if unset.
            self.coords = np.asanyarray(coords)

        self.plotting = {
            "vertex_size": 100,
            "vertex_color": (0.12, 0.47, 0.71, 0.5),
            "edge_color": (0.5, 0.5, 0.5, 0.5),
            "edge_width": 2,
            "edge_style": "-",
            "highlight_color": "C1",
            "normalize_intercept": 0.25,
        }
        self.plotting.update(plotting)
        self.signals = dict()

        # Attributes that are lazily computed.
        self._A = None
        self._d = None
        self._dw = None
        self._lmax = None
        self._lmax_method = None
        self._U = None
        self._e = None
        self._coherence = None
        self._D = None
        # self._L = None

        # TODO: what about Laplacian? Lazy as Fourier, or disallow change?
        self.lap_type = lap_type
        self.compute_laplacian(lap_type)

        # TODO: kept for backward compatibility.
        self.Ne = self.n_edges
        self.N = self.n_vertices

    def _get_extra_repr(self):
        return dict()

    def __repr__(self, limit=None):
        s = ""
        for attr in ["n_vertices", "n_edges"]:
            s += f"{attr}={getattr(self, attr)}, "
        for i, (key, value) in enumerate(self._get_extra_repr().items()):
            if (limit is not None) and (i == limit - 2):
                s += "..., "
                break
            s += f"{key}={value}, "
        return f"{self.__class__.__name__}({s[:-2]})"

    def set_signal(self, signal, name):
        r"""Attach a signal to the graph.

        Attached signals can be accessed (and modified or deleted) through the
        :attr:`signals` dictionary.

        Parameters
        ----------
        signal : array_like
            A sequence that assigns a value to each vertex.
            The value of the signal at vertex `i` is ``signal[i]``.
        name : String
            Name of the signal used as a key in the :attr:`signals` dictionary.

        Examples
        --------
        >>> graph = graphs.Sensor(10)
        >>> signal = np.arange(graph.n_vertices)
        >>> graph.set_signal(signal, 'mysignal')
        >>> graph.signals
        {'mysignal': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}

        """
        signal = self._check_signal(signal)
        self.signals[name] = signal

    def subgraph(self, vertices):
        r"""Create a subgraph from a list of vertices.

        Parameters
        ----------
        vertices : list
            Vertices to keep.
            Either a list of indices or an indicator function.

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
        graph = Graph(adjacency, self.lap_type, coords, self.plotting)
        for name, signal in self.signals.items():
            graph.set_signal(signal[vertices], name)
        return graph

    def is_weighted(self):
        r"""Check if the graph is weighted.

        A graph is unweighted (binary) if and only if all the entries in the
        adjacency matrix are either zero or one.

        Returns
        -------
        weighted : bool
            True if the graph is weighted, False otherwise.

        Examples
        --------

        Unweighted (binary) graph:

        >>> graph = graphs.Graph([
        ...     [0, 1, 0],
        ...     [1, 0, 1],
        ...     [0, 1, 0],
        ... ])
        >>> graph.is_weighted()
        False

        Weighted graph:

        >>> graph = graphs.Graph([
        ...     [0, 2, 0],
        ...     [2, 0, 1],
        ...     [0, 1, 0],
        ... ])
        >>> graph.is_weighted()
        True

        """
        return not np.all(self.W.data == 1)

    def is_connected(self):
        r"""Check if the graph is connected (cached).

        A graph is connected if and only if there exists a (directed) path
        between any two vertices.

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
        if self._connected is not None:
            return self._connected

        adjacencies = [self.W]
        if self.is_directed():
            adjacencies.append(self.W.T)

        for adjacency in adjacencies:
            visited = np.zeros(self.n_vertices, dtype=bool)
            stack = {0}

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

    def is_directed(self):
        r"""Check if the graph has directed edges (cached).

        In this framework, we consider that a graph is directed if and
        only if its weight matrix is not symmetric.

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
        if self._directed is None:
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
            self.logger.error(
                "Inconsistent shape to extract components. " "Square matrix required."
            )
            return None

        if self.is_directed():
            raise NotImplementedError("Directed graphs not supported yet.")

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
                    stack.update(
                        {idx for idx in self.A[v, :].nonzero()[1] if not visited[idx]}
                    )

            comp = sorted(comp)
            self.logger.info(
                ("Constructing subgraph for component of " "size {}.").format(len(comp))
            )
            G = self.subgraph(comp)
            G.info = {"orig_idx": comp}
            graphs.append(G)

        return graphs

    def compute_laplacian(self, lap_type="combinatorial"):
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
        >>> G.compute_fourier_basis()
        >>> -1e-10 < G.e[0] < 1e-10 < G.e[-1] < 2
        True

        """

        if lap_type != self.lap_type:
            # Those attributes are invalidated when the Laplacian is changed.
            # Alternative: don't allow the user to change the Laplacian.
            self._lmax = None
            self._U = None
            self._e = None
            self._coherence = None
            self._D = None

        self.lap_type = lap_type

        if not self.is_directed():
            W = self.W
        else:
            W = utils.symmetrize(self.W, method="average")

        if lap_type == "combinatorial":
            D = sparse.diags(self.dw)
            self.L = D - W
        elif lap_type == "normalized":
            d = np.zeros(self.n_vertices)
            disconnected = self.dw == 0
            np.power(self.dw, -0.5, where=~disconnected, out=d)
            D = sparse.diags(d)
            self.L = sparse.identity(self.n_vertices) - D * W * D
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        else:
            raise ValueError(f"Unknown Laplacian type {lap_type}")

    def _check_signal(self, s):
        r"""Check if signal is valid."""
        s = np.asanyarray(s)
        if s.shape[0] != self.n_vertices:
            raise ValueError(
                "First dimension must be the number of vertices "
                "G.N = {}, got {}.".format(self.N, s.shape)
            )
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

        See Also
        --------
        grad : compute the gradient of a vertex signal

        Examples
        --------

        Non-directed graph:

        >>> graph = graphs.Path(5, directed=False)
        >>> signal = [0, 2, 2, 4, 4]
        >>> graph.dirichlet_energy(signal)
        8.0
        >>> # The Dirichlet energy is indeed the squared norm of the gradient.
        >>> graph.compute_differential_operator()
        >>> graph.grad(signal)
        array([2., 0., 2., 0.])

        Directed graph:

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
    def W(self):
        r"""Weighted adjacency matrix of the graph."""
        return self._adjacency

    @W.setter
    def W(self, value):
        # TODO: user can still do G.W[0, 0] = 1, or modify the passed W.
        raise AttributeError(
            "In-place modification of the graph is not "
            "supported. Create another Graph object."
        )

    @property
    def A(self):
        r"""Graph adjacency matrix (the binary version of W).

        The adjacency matrix defines which edges exist on the graph.
        It is represented as an N-by-N matrix of booleans.
        :math:`A_{i,j}` is True if :math:`W_{i,j} > 0`.
        """
        if self._A is None:
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
        if self._d is None:
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
        if self._dw is None:
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
        if self._lmax is None:
            self.logger.warning(
                "The largest eigenvalue G.lmax is not "
                "available, we need to estimate it. "
                "Explicitly call G.estimate_lmax() or "
                "G.compute_fourier_basis() "
                "once beforehand to suppress the warning."
            )
            self.estimate_lmax()
        return self._lmax

    def estimate_lmax(self, method="lanczos"):
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
        >>> G.estimate_lmax(method='lanczos')  # Estimate.
        >>> print('{:.2f}'.format(G.lmax))
        13.92
        >>> G.estimate_lmax(method='bounds')  # Upper bound.
        >>> print('{:.2f}'.format(G.lmax))
        18.58

        """
        if method == self._lmax_method:
            return
        self._lmax_method = method

        if method == "lanczos":
            try:
                # We need to cast the matrix L to a supported type.
                # TODO: not good for memory. Cast earlier?
                lmax = sparse.linalg.eigsh(
                    self.L.asfptype(),
                    k=1,
                    tol=5e-3,
                    ncv=min(self.N, 10),
                    return_eigenvectors=False,
                )
                lmax = lmax[0]
                assert lmax <= self._get_upper_bound() + 1e-12
                lmax *= 1.01  # Increase by 1% to be robust to errors.
                self._lmax = lmax
            except sparse.linalg.ArpackNoConvergence:
                raise ValueError(
                    "The Lanczos method did not converge. " "Try to use bounds."
                )

        elif method == "bounds":
            self._lmax = self._get_upper_bound()

        else:
            raise ValueError(f"Unknown method {method}")

    def _get_upper_bound(self):
        r"""Return an upper bound on the eigenvalues of the Laplacian."""

        if self.lap_type == "normalized":
            return 2  # Equal iff the graph is bipartite.
        elif self.lap_type == "combinatorial":
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
                W = utils.symmetrize(self.W, method="average")
            m = W.dot(self.dw) / self.dw  # Mean degree of adjacent vertices.
            bounds += [np.max(self.dw + m)]
            # Good review: On upper bounds for Laplacian graph eigenvalues.
            return min(bounds)
        else:
            raise ValueError("Unknown Laplacian type " "{}".format(self.lap_type))

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
            W = sparse.triu(self.W, format="coo")

        sources = W.row
        targets = W.col
        weights = W.data

        assert self.n_edges == sources.size == targets.size == weights.size
        return sources, targets, weights

    def plot(
        self,
        vertex_color=None,
        vertex_size=None,
        highlight=[],
        edges=None,
        edge_color=None,
        edge_width=None,
        indices=False,
        colorbar=True,
        limits=None,
        ax=None,
        title=None,
        backend=None,
    ):
        r"""Docstring overloaded at import time."""
        from pygsp.plotting import _plot_graph

        return _plot_graph(
            self,
            vertex_color=vertex_color,
            vertex_size=vertex_size,
            highlight=highlight,
            edges=edges,
            indices=indices,
            colorbar=colorbar,
            edge_color=edge_color,
            edge_width=edge_width,
            limits=limits,
            ax=ax,
            title=title,
            backend=backend,
        )

    def plot_signal(self, *args, **kwargs):
        r"""Deprecated, use plot() instead."""
        return self.plot(*args, **kwargs)

    def plot_spectrogram(self, node_idx=None):
        r"""Docstring overloaded at import time."""
        from pygsp.plotting import _plot_spectrogram

        _plot_spectrogram(self, node_idx=node_idx)
