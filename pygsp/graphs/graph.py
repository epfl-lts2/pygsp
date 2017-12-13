# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
from scipy import sparse

from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5


class Graph(fourier.GraphFourier, difference.GraphDifference):
    r"""Base graph class.

    * Provide a common interface (and implementation) to graph objects.
    * Can be instantiated to construct custom graphs from a weight matrix.
    * Initialize attributes for derived classes.

    Parameters
    ----------
    W : sparse matrix or ndarray
        The weight matrix which encodes the graph.
    gtype : string
        Graph type, a free-form string to help us recognize the kind of graph
        we are dealing with (default is 'unknown').
    lap_type : 'combinatorial', 'normalized'
        The type of Laplacian to be computed by :func:`compute_laplacian`
        (default is 'combinatorial').
    coords : ndarray
        Vertices coordinates (default is None).
    plotting : dict
        Plotting parameters.

    Attributes
    ----------
    N : int
        the number of nodes / vertices in the graph.
    Ne : int
        the number of edges / links in the graph, i.e. connections between
        nodes.
    W : sparse matrix
        the weight matrix which contains the weights of the connections.
        It is represented as an N-by-N matrix of floats.
        :math:`W_{i,j} = 0` means that there is no direct connection from
        i to j.
    gtype : string
        the graph type is a short description of the graph object designed to
        help sorting the graphs.
    L : sparse matrix
        the graph Laplacian, an N-by-N matrix computed from W.
    lap_type : 'normalized', 'combinatorial'
        the kind of Laplacian that was computed by :func:`compute_laplacian`.
    coords : ndarray
        vertices coordinates in 2D or 3D space. Used for plotting only. Default
        is None.
    plotting : dict
        plotting parameters.

    Examples
    --------
    >>> W = np.arange(4).reshape(2, 2)
    >>> G = graphs.Graph(W)

    """

    def __init__(self, W, gtype='unknown', lap_type='combinatorial',
                 coords=None, plotting={}):

        self.logger = utils.build_logger(__name__)

        if len(W.shape) != 2 or W.shape[0] != W.shape[1]:
            raise ValueError('W has incorrect shape {}'.format(W.shape))

        # Don't keep edges of 0 weight. Otherwise Ne will not correspond to the
        # real number of edges. Problematic when e.g. plotting.
        W = sparse.csr_matrix(W)
        W.eliminate_zeros()

        self.N = W.shape[0]
        self.W = sparse.lil_matrix(W)

        # Don't count edges two times if undirected.
        # Be consistent with the size of the differential operator.
        if self.is_directed():
            self.Ne = self.W.nnz
        else:
            self.Ne = sparse.tril(W).nnz

        self.check_weights()

        self.gtype = gtype

        self.compute_laplacian(lap_type)

        if coords is not None:
            self.coords = coords

        self.plotting = {'vertex_size': 100,
                         'vertex_color': (0.12, 0.47, 0.71, 1),
                         'edge_color': (0.5, 0.5, 0.5, 1),
                         'edge_width': 1,
                         'edge_style': '-'}
        self.plotting.update(plotting)

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
        kind : string or array-like
            Kind of coordinates to generate. It controls the position of the
            nodes when plotting the graph. Can either pass an array of size Nx2
            or Nx3 to set the coordinates manually or the name of a layout
            algorithm. Available algorithms: community2D, random2D, random3D,
            ring2D, line1D, spring. Default is 'spring'.
        kwargs : dict
            Additional parameters to be passed to the Fruchterman-Reingold
            force-directed algorithm when kind is spring.

        Examples
        --------
        >>> G = graphs.ErdosRenyi()
        >>> G.set_coordinates()
        >>> G.plot()

        """

        if not isinstance(kind, str):
            coords = np.asarray(kind).squeeze()
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

        else:
            raise ValueError('Unexpected argument king={}.'.format(kind))

    def subgraph(self, ind):
        r"""Create a subgraph given indices.

        Parameters
        ----------
        ind : list
            Nodes to keep

        Returns
        -------
        sub_G : Graph
            Subgraph

        Examples
        --------
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> ind = [1, 3]
        >>> sub_G = G.subgraph(ind)

        """
        if not isinstance(ind, list) and not isinstance(ind, np.ndarray):
            raise TypeError('The indices must be a list or a ndarray.')

        # N = len(ind) # Assigned but never used

        sub_W = self.W.tocsr()[ind, :].tocsc()[:, ind]
        return Graph(sub_W, gtype="sub-{}".format(self.gtype))

    def is_connected(self, recompute=False):
        r"""Check the strong connectivity of the graph (cached).

        It uses DFS travelling on graph to ensure that each node is visited.
        For undirected graphs, starting at any vertex and trying to access all
        others is enough.
        For directed graphs, one needs to check that a random vertex is
        accessible by all others
        and can access all others. Thus, we can transpose the adjacency matrix
        and compute again with the same starting point in both phases.

        Parameters
        ----------
        recompute: bool
            Force to recompute the connectivity if already known.

        Returns
        -------
        connected : bool
            True if the graph is connected.

        Examples
        --------
        >>> from scipy import sparse
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> connected = G.is_connected()

        """
        if hasattr(self, '_connected') and not recompute:
            return self._connected

        if self.is_directed(recompute=recompute):
            adj_matrices = [self.A, self.A.T]
        else:
            adj_matrices = [self.A]

        for adj_matrix in adj_matrices:
            visited = np.zeros(self.A.shape[0], dtype=bool)
            stack = set([0])

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True

                    # Add indices of nodes not visited yet and accessible from
                    # v
                    stack.update(set([idx
                                      for idx in adj_matrix[v, :].nonzero()[1]
                                      if not visited[idx]]))

            if not visited.all():
                self._connected = False
                return self._connected

        self._connected = True
        return self._connected

    def is_directed(self, recompute=False):
        r"""Check if the graph has directed edges (cached).

        In this framework, we consider that a graph is directed if and
        only if its weight matrix is non symmetric.

        Parameters
        ----------
        recompute : bool
            Force to recompute the directedness if already known.

        Returns
        -------
        directed : bool
            True if the graph is directed.

        Notes
        -----
        Can also be used to check if a matrix is symmetrical

        Examples
        --------
        >>> from scipy import sparse
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> directed = G.is_directed()

        """
        if hasattr(self, '_directed') and not recompute:
            return self._directed

        self._directed = np.abs(self.W - self.W.T).sum() != 0
        return self._directed

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
        >>> G = graphs.Graph(W=W)
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
            stack = set(np.nonzero(~visited)[0])
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

        The result is accessible by the L attribute.

        Parameters
        ----------
        lap_type : 'combinatorial', 'normalized'
            The type of Laplacian to compute. Default is combinatorial.

        Notes
        -----
        For undirected graphs, the combinatorial Laplacian is defined as

        .. math:: L = D - W,

        where :math:`W` is the weight matrix and :math:`D` the degree matrix,
        and the normalized Laplacian is defined as

        .. math:: L = I - D^{-1/2} W D^{-1/2},

        where :math:`I` is the identity matrix.

        Examples
        --------
        >>> G = graphs.Sensor(50)
        >>> G.L.shape
        (50, 50)
        >>>
        >>> G.compute_laplacian('combinatorial')
        >>> G.compute_fourier_basis()
        >>> -1e-10 < G.e[0] < 1e-10  # Smallest eigenvalue close to 0.
        True
        >>>
        >>> G.compute_laplacian('normalized')
        >>> G.compute_fourier_basis(recompute=True)
        >>> -1e-10 < G.e[0] < 1e-10 < G.e[-1] < 2  # Spectrum in [0, 2].
        True

        """

        if lap_type not in ['combinatorial', 'normalized']:
            raise ValueError('Unknown Laplacian type {}'.format(lap_type))
        self.lap_type = lap_type

        if self.is_directed():

            if lap_type == 'combinatorial':
                D1 = sparse.diags(np.ravel(self.W.sum(0)), 0)
                D2 = sparse.diags(np.ravel(self.W.sum(1)), 0)
                self.L = 0.5 * (D1 + D2 - self.W - self.W.T).tocsc()

            elif lap_type == 'normalized':
                raise NotImplementedError('Directed graphs with normalized '
                                          'Laplacian not supported yet.')

        else:

            if lap_type == 'combinatorial':
                D = sparse.diags(np.ravel(self.W.sum(1)), 0)
                self.L = (D - self.W).tocsc()

            elif lap_type == 'normalized':
                d = np.power(self.dw, -0.5)
                D = sparse.diags(np.ravel(d), 0).tocsc()
                self.L = sparse.identity(self.N) - D * self.W * D


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
        r"""The degree (the number of neighbors) of each node."""
        if not hasattr(self, '_d'):
            self._d = np.asarray(self.A.sum(axis=1)).squeeze()
        return self._d

    @property
    def dw(self):
        r"""The weighted degree (the sum of weighted edges) of each node."""
        if not hasattr(self, '_dw'):
            self._dw = np.asarray(self.W.sum(axis=1)).squeeze()
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

    def estimate_lmax(self, recompute=False):
        r"""Estimate the Laplacian's largest eigenvalue (cached).

        The result is cached and accessible by the :attr:`lmax` property.

        Exact value given by the eigendecomposition of the Laplacian, see
        :func:`compute_fourier_basis`. That estimation is much faster than the
        eigendecomposition.

        Parameters
        ----------
        recompute : boolean
            Force to recompute the largest eigenvalue. Default is false.

        Notes
        -----
        Runs the implicitly restarted Lanczos method with a large tolerance,
        then increases the calculated largest eigenvalue by 1 percent. For much
        of the PyGSP machinery, we need to approximate wavelet kernels on an
        interval that contains the spectrum of L. The only cost of using a
        larger interval is that the polynomial approximation over the larger
        interval may be a slightly worse approximation on the actual spectrum.
        As this is a very mild effect, it is not necessary to obtain very tight
        bounds on the spectrum of L.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> print('{:.2f}'.format(G.lmax))
        13.78
        >>> G = graphs.Logo()
        >>> G.estimate_lmax(recompute=True)
        >>> print('{:.2f}'.format(G.lmax))
        13.92

        """
        if hasattr(self, '_lmax') and not recompute:
            return

        try:
            lmax = sparse.linalg.eigsh(self.L, k=1, tol=5e-3,
                                       ncv=min(self.N, 10),
                                       return_eigenvectors=False)
            lmax = lmax[0]
            lmax *= 1.01  # Increase by 1 percent to be robust to errors.

        except sparse.linalg.ArpackNoConvergence:
            self.logger.warning('Lanczos method did not converge. '
                                'Using an alternative method.')
            if self.lap_type == 'normalized':
                lmax = 2  # Spectrum is bounded by [0, 2].
            elif self.lap_type == 'combinatorial':
                lmax = 2 * np.max(self.dw)
            else:
                raise ValueError('Unknown Laplacian type '
                                 '{}'.format(self.lap_type))

        self._lmax = lmax

    def get_edge_list(self):
        r"""Return an edge list, an alternative representation of the graph.

        The weighted adjacency matrix is the canonical form used in this
        package to represent a graph as it is the easiest to work with when
        considering spectral methods.

        Returns
        -------
        v_in : vector of int
        v_out : vector of int
        weights : vector of float

        Examples
        --------
        >>> G = graphs.Logo()
        >>> v_in, v_out, weights = G.get_edge_list()
        >>> v_in.shape, v_out.shape, weights.shape
        ((3131,), (3131,), (3131,))

        """

        if self.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        else:
            v_in, v_out = sparse.tril(self.W).nonzero()
            weights = self.W[v_in, v_out]
            weights = weights.toarray().squeeze()

            # TODO G.ind_edges = sub2ind(size(G.W), G.v_in, G.v_out)

            assert self.Ne == v_in.size == v_out.size == weights.size
            return v_in, v_out, weights

    def modulate(self, f, k):
        r"""Modulate the signal *f* to the frequency *k*.

        Parameters
        ----------
        f : ndarray
            Signal (column)
        k : int
            Index of frequencies

        Returns
        -------
        fm : ndarray
            Modulated signal

        """

        nt = np.shape(f)[1]
        fm = np.kron(np.ones((1, nt)), self.U[:, k])
        fm *= np.kron(np.ones((nt, 1)), f)
        fm *= np.sqrt(self.N)
        return fm

    def plot(self, **kwargs):
        r"""Plot the graph.

        See :func:`pygsp.plotting.plot_graph`.
        """
        from pygsp import plotting
        plotting.plot_graph(self, **kwargs)

    def plot_signal(self, signal, **kwargs):
        r"""Plot a signal on that graph.

        See :func:`pygsp.plotting.plot_signal`.
        """
        from pygsp import plotting
        plotting.plot_signal(self, signal, **kwargs)

    def plot_spectrogram(self, **kwargs):
        r"""Plot the graph's spectrogram.

        See :func:`pygsp.plotting.plot_spectrogram`.
        """
        from pygsp import plotting
        plotting.plot_spectrogram(self, **kwargs)

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
                pos_arr[i] = np.asarray(pos[i])

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
            Ai = np.asarray(A[i, :].toarray())
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
