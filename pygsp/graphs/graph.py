# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
from scipy import sparse
from scipy.linalg import svd

from pygsp.utils import build_logger


class Graph(object):
    r"""
    The base graph class.

    * Provide a common interface (and implementation) to graph objects.
    * Can be instantiated to construct custom graphs from a weight matrix.
    * Initialize attributes for derived classes.

    Parameters
    ----------
    W : sparse matrix or ndarray
        weight matrix which encodes the graph
    gtype : string
        graph type (default is 'unknown')
    lap_type : 'none', 'normalized', 'combinatorial'
        Laplacian type (default is 'combinatorial')
    coords : ndarray
        vertices coordinates (default is None)
    plotting : dict
        plotting parameters
    perform_checks : bool
        Whether to check if the graph is connected. Warn if not.

    Attributes
    ----------
    N : int
        the number of nodes / vertices in the graph.
    Ne : int
        the number of edges / links in the graph, i.e. connections between
        nodes.
    W : ndarray
        the weight matrix which contains the weights of the connections.
        It is represented as an N-by-N matrix of floats.
        :math:`W_{i,j} = 0` means that there is no direct connection from
        i to j.
    A : sparse matrix or ndarray
        the adjacency matrix defines which edges exist on the graph.
        It is represented as an N-by-N matrix of booleans.
        :math:`A_{i,j}` is True if :math:`W_{i,j} > 0`.
    d : ndarray
        the degree vector is a vector of length N which represents the number
        of edges connected to each node.
    gtype : string
        the graph type is a short description of the graph object designed to
        help sorting the graphs.
    L : sparse matrix or ndarray
        the graph Laplacian, an N-by-N matrix computed from W.
    lap_type : 'none', 'normalized', 'combinatorial'
        determines which kind of Laplacian will be computed by
        :func:`create_laplacian`.
    coords : ndarray
        vertices coordinates in 2D or 3D space. Used for plotting only. Default
        is None.
    plotting : dict
        plotting parameters.

    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> W = np.arange(4).reshape(2, 2)
    >>> G = graphs.Graph(W)

    """

    def __init__(self, W, gtype='unknown', lap_type='combinatorial',
                 coords=None, plotting={}, perform_checks=True, **kwargs):

        self.logger = build_logger(__name__, **kwargs)

        if len(W.shape) != 2 or W.shape[0] != W.shape[1]:
            self.logger.error('W has incorrect shape {}'.format(W.shape))

        self.N = W.shape[0]
        self.W = sparse.lil_matrix(W)
        self.check_weights()

        self.A = self.W > 0
        self.Ne = self.W.nnz
        self.d = self.A.sum(axis=1)
        self.gtype = gtype
        self.lap_type = lap_type

        if coords is not None:
            self.coords = coords

        # Very expensive for big graphs. Allow user to opt out.
        if perform_checks:
            if not self.is_connected():
                self.logger.warning('Graph is not connected!')

        self.create_laplacian(lap_type)

        self.plotting = {'vertex_size': 10, 'edge_width': 1,
                         'edge_style': '-', 'vertex_color': 'b'}
        self.plotting.update(plotting)

    def check_weights(self):
        r"""
        Check the characteristics of the weights matrix.

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
        >>> import numpy as np
        >>> from pygsp import graphs
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
            self.logger.warning("GSP_TEST_WEIGHTS: There is an infinite "
                                "value in the weight matrix")
            has_inf_val = True

        if abs(self.W.diagonal()).sum() != 0:
            self.logger.warning("GSP_TEST_WEIGHTS: The main diagonal of "
                                "the weight matrix is not 0!")
            diag_is_not_zero = True

        if self.W.get_shape()[0] != self.W.get_shape()[1]:
            self.logger.warning("GSP_TEST_WEIGHTS: The weight matrix is "
                                "not square!")
            is_not_square = True

        if np.isnan(self.W.sum()):
            self.logger.warning("GSP_TEST_WEIGHTS: There is an NaN "
                                "value in the weight matrix")
            has_nan_value = True

        return {'has_inf_val': has_inf_val,
                'has_nan_value': has_nan_value,
                'is_not_square': is_not_square,
                'diag_is_not_zero': diag_is_not_zero}

    def update_graph_attr(self, *args, **kwargs):
        r"""
        Recompute some attribute of the graph.

        Parameters
        ----------
        args: list of string
            the arguments that will be not changed and not re-compute.
        kwargs: Dictionnary
            The arguments with their new value.

        Returns
        -------
        The same Graph with some updated values.

        Notes
        -----
        This method is usefull if you want to give a new weight matrix
        (W) and compute the adjacency matrix (A) and more again.
        The valid attributes are ['W', 'A', 'N', 'd', 'Ne', 'gtype',
        'directed', 'coords', 'lap_type', 'L', 'plotting']

        Examples
        --------
        >>> from pygsp import graphs
        >>> G = graphs.Ring(N=10)
        >>> newW = G.W
        >>> newW[1] = 1
        >>> G.update_graph_attr('N', 'd', W=newW)

        Updates all attributes of G except 'N' and 'd'

        """
        graph_attr = {}
        valid_attributes = ['W', 'A', 'N', 'd', 'Ne', 'gtype', 'directed',
                            'coords', 'lap_type', 'L', 'plotting']

        for i in args:
            if i in valid_attributes:
                graph_attr[i] = getattr(self, i)
            else:
                self.logger.warning(('Your attribute {} does not figure in '
                                     'the valid attributes, which are '
                                     '{}').format(i, valid_attributes))

        for i in kwargs:
            if i in valid_attributes:
                if i in graph_attr:
                    self.logger.info('You already gave this attribute as '
                                     'an argument. Therefore, it will not '
                                     'be recomputed.')
                else:
                    graph_attr[i] = kwargs[i]
            else:
                self.logger.warning(('Your attribute {} does not figure in '
                                     'the valid attributes, which are '
                                     '{}').format(i, valid_attributes))

        from pygsp.graphs import NNGraph
        if isinstance(self, NNGraph):
            super(NNGraph, self).__init__(**graph_attr)
        else:
            super(type(self), self).__init__(**graph_attr)

    def copy_graph_attributes(self, Gn, ctype=True):
        r"""
        Copy some parameters of the graph into a given one.

        Parameters
        ----------:
        G : Graph structure
        ctype : bool
            Flag to select what to copy (Default is True)
        Gn : Graph structure
            The graph where the parameters will be copied

        Returns
        -------
        Gn : Partial graph structure

        Examples
        --------
        >>> from pygsp import graphs
        >>> Torus = graphs.Torus()
        >>> G = graphs.TwoMoons()
        >>> G.copy_graph_attributes(ctype=False, Gn=Torus);

        """
        if hasattr(self, 'plotting'):
            Gn.plotting = self.plotting

        if ctype:
            if hasattr(self, 'coords'):
                Gn.coords = self.coords
        else:
            if hasattr(Gn.plotting, 'limits'):
                del Gn.plotting['limits']

        if hasattr(self, 'lap_type'):
            Gn.lap_type = self.lap_type
            Gn.create_laplacian()

    def set_coords(self, kind='spring', **kwargs):
        r"""
        Set coordinates for the vertices.

        Parameters
        ----------
        kind : string
            The kind of display. Default is 'spring'.
            Accepting ['community2D', 'manual', 'random2D', 'random3D',
            'ring2D', 'spring'].
        coords : np.ndarray
            An array of coordinates in 2D or 3D. Used only if kind is manual.
            Set the coordinates to this array as is.

        Examples
        --------
        >>> from pygsp import graphs
        >>> G = graphs.ErdosRenyi()
        >>> G.set_coords()
        >>> G.plot()

        """
        if kind not in ['community2D', 'manual', 'random2D', 'random3D',
                        'ring2D', 'spring']:
            raise ValueError('Unexpected kind argument. Got {}.'.format(kind))

        if kind == 'manual':
            coords = kwargs.pop('coords', None)
            if isinstance(coords, list):
                coords = np.array(coords)
            if isinstance(coords, np.ndarray) and len(coords.shape) == 2 and \
                    coords.shape[0] == self.N and 2 <= coords.shape[1] <= 3:
                self.coords = coords
            else:
                raise ValueError('Expecting coords to be a list or ndarray '
                                 'of size Nx2 or Nx3.')

        elif kind == 'ring2D':
            tmp = np.arange(self.N).reshape(self.N, 1)
            self.coords = np.concatenate((np.cos(tmp * 2 * np.pi / self.N),
                                          np.sin(tmp * 2 * np.pi / self.N)),
                                         axis=1)

        elif kind == 'random2D':
            self.coords = np.random.rand(self.N, 2)

        elif kind == 'random3D':
            self.coords = np.random.rand(self.N, 3)

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

    def subgraph(self, ind):
        r"""
        Create a subgraph from G keeping only the given indices.

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
        >>> from pygsp import graphs
        >>> import numpy as np
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
        r"""
        Check the strong connectivity of the input graph. Result is cached.

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
        >>> from pygsp import graphs
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> connected = G.is_connected()

        """
        if hasattr(self, '_connected') and not recompute:
            return self._connected

        if self.A.shape[0] != self.A.shape[1]:
            self.logger.error("Inconsistent shape to test connectedness. "
                              "Set to False.")
            self._connected = False
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
        r"""
        Check if the graph has directed edges. Result is cached.

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
        >>> from pygsp import graphs
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> directed = G.is_directed()

        """
        if hasattr(self, '_directed') and not recompute:
            return self._directed

        if np.diff(np.shape(self.W))[0]:
            raise ValueError("Matrix dimensions mismatch, expecting square "
                             "matrix.")

        self._directed = np.abs(self.W - self.W.T).sum() != 0
        return self._directed

    def extract_components(self):
        r"""
        Split the graph into several connected components.

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
        >>> import pygsp
        >>> W = sparse.rand(10, 10, 0.2)
        >>> W = pygsp.utils.symmetrize(W)
        >>> G = pygsp.graphs.Graph(W=W)
        >>> components = G.extract_components()
        >>> has_sinks = 'sink' in components[0].info
        >>> sinks_0 = components[0].info['sink'] if has_sinks else []

        """
        if self.A.shape[0] != self.A.shape[1]:
            self.logger.error('Inconsistent shape to extract components. '
                              'Square matrix required.')
            return None

        if self.is_directed():
            raise NotImplementedError('Focusing on undirected graphs first.')

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

    def compute_fourier_basis(self, smallest_first=True, recompute=False,
                              **kwargs):
        r"""
        Compute the fourier basis of the graph.

        Parameters
        ----------
        smallest_first: bool
            Define the order of the eigenvalues.
            Default is smallest first (True).
        recompute: bool
            Force to recompute the Fourier basis if already existing.

        Notes
        -----
        'G.compute_fourier_basis()' computes a full eigendecomposition of
        the graph Laplacian G.L:

        .. math:: {\cal L} = U \Lambda U^*

        where :math:`\Lambda` is a diagonal matrix of eigenvalues.

        *G.e* is a column vector of length *G.N* containing the Laplacian
        eigenvalues. The largest eigenvalue is stored in *G.lmax*.
        The eigenvectors are stored as column vectors of *G.U* in the same
        order that the eigenvalues. Finally, the coherence of the
        Fourier basis is in *G.mu*.

        References
        ----------
        See :cite:`chung1997spectral`

        Examples
        --------
        >>> from pygsp import graphs
        >>> N = 50
        >>> G = graphs.Sensor(N)
        >>> G.compute_fourier_basis()

        """
        if hasattr(self, 'e') and hasattr(self, 'U') and not recompute:
            return

        if self.N > 3000:
            self.logger.warning("Performing full eigendecomposition of a "
                                "large matrix may take some time.")

        if not hasattr(self, 'L'):
            raise AttributeError("Graph Laplacian is missing.")

        eigenvectors, eigenvalues, _ = svd(self.L.todense())

        inds = np.argsort(eigenvalues)
        if not smallest_first:
            inds = inds[::-1]

        self.e = np.sort(eigenvalues)
        self.lmax = np.max(self.e)
        self.U = eigenvectors[:, inds]
        self.mu = np.max(np.abs(self.U))

    def create_laplacian(self, lap_type='combinatorial'):
        r"""
        Create a new graph laplacian.

        Parameters
        ----------
        lap_type : string
            The laplacian type to use. Default is 'combinatorial'. Other
            possible values are 'none' and 'normalized', which are not yet
            implemented for directed graphs.

        """
        if self.W.shape == (1, 1):
            self.L = sparse.lil_matrix(0)
            return

        if lap_type not in ['combinatorial', 'normalized', 'none']:
            raise AttributeError('Unknown laplacian type {}'.format(lap_type))
        self.lap_type = lap_type

        if self.is_directed():
            if lap_type == 'combinatorial':
                L = 0.5 * (sparse.diags(np.ravel(self.W.sum(0)), 0) +
                           sparse.diags(np.ravel(self.W.sum(1)), 0) -
                           self.W - self.W.T).tocsc()
            elif lap_type == 'normalized':
                raise NotImplementedError('Yet. Ask Nathanael.')
            elif lap_type == 'none':
                L = sparse.lil_matrix(0)

        else:
            if lap_type == 'combinatorial':
                L = (sparse.diags(np.ravel(self.W.sum(1)), 0) - self.W).tocsc()
            elif lap_type == 'normalized':
                D = sparse.diags(
                    np.ravel(np.power(self.W.sum(1), -0.5)), 0).tocsc()
                L = sparse.identity(self.N) - D * self.W * D
            elif lap_type == 'none':
                L = sparse.lil_matrix(0)

        self.L = L

    def estimate_lmax(self, recompute=False):
        r"""
        Estimate the maximal eigenvalue.

        Exact value given by the eigendecomposition of the Laplacia, see
        :func:`compute_fourier_basis`.

        Parameters
        ----------
        recompute : boolean
            Force to recompute the maximal eigenvalue. Default is false.

        Returns
        -------
        lmax : float
            An estimation of the largest eigenvalue.

        Examples
        --------
        >>> from pygsp import graphs
        >>> import numpy as np
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> print('{:.2f}'.format(G.estimate_lmax()))
        41.59

        """
        if hasattr(self, 'lmax') and not recompute:
            return self.lmax

        try:
            # For robustness purposes, increase the error by 1 percent
            lmax = 1.01 * \
                sparse.linalg.eigs(self.L, k=1, tol=5e-3, ncv=10)[0][0]

        except sparse.linalg.ArpackNoConvergence:
            self.logger.warning('GSP_ESTIMATE_LMAX: '
                                'Cannot use default method.')
            lmax = 2. * np.max(self.d)

        lmax = np.real(lmax)
        self.lmax = lmax.sum()
        return self.lmax

    def plot(self, **kwargs):
        r"""
        Plot the graph.

        See :func:`pygsp.plotting.plot_graph`.
        """
        from pygsp import plotting
        plotting.plot_graph(self, **kwargs)

    def plot_signal(self, signal, **kwargs):
        r"""
        Plot a signal on that graph.

        See :func:`pygsp.plotting.plot_signal`.
        """
        from pygsp import plotting
        plotting.plot_signal(self, signal, **kwargs)

    def show_spectrogram(self, **kwargs):
        r"""
        Plot the spectrogram for the graph object.

        See :func:`pygsp.plotting.plot_spectrogram`.
        """
        from pygsp import plotting
        plotting.plot_spectrogram(self, **kwargs)

    def _fruchterman_reingold_layout(self, dim=2, k=None, pos=None, fixed=[],
                                     iterations=50, scale=1.0, center=None):
        # TODO doc
        # Position nodes using Fruchterman-Reingold force-directed algorithm.

        if center is None:
            center = np.zeros((1, dim))

        if np.shape(center)[1] != dim:
            self.logger.error('Spring coordinates : center has wrong size.')
            center = np.zeros((1, dim))

        dom_size = 1.
        if pos is not None:
            # Determine size of existing domain to adjust initial positions
            dom_size = np.max(pos)
            shape = (self.N, dim)
            pos_arr = np.random.random(shape) * dom_size + center
            for i in range(self.N):
                pos_arr[i] = np.asarray(pos[i])
        else:
            pos_arr = None

        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts that are not near 1x1
            k = dom_size / np.sqrt(self.N)
        pos = _sparse_fruchterman_reingold(
            self.A, dim, k, pos_arr, fixed, iterations)

        if fixed is None:
            pos = _rescale_layout(pos, scale=scale) + center
        return pos


def _sparse_fruchterman_reingold(A, dim=2, k=None, pos=None, fixed=None,
                                 iterations=50):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    nnodes = A.shape[0]

    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except Exception:
        A = (sparse.coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.random.random((nnodes, dim))

    # no fixed nodes
    if fixed is None:
        fixed = []

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
