# -*- coding: utf-8 -*-

from pygsp.utils import build_logger

import numpy as np
from scipy import sparse

from collections import Counter


class Graph(object):
    r"""
    The main graph object.

    It is used to initialize by default every missing field of the subclass graphs.
    It can also be used alone to initialize customs graphs.

    Parameters
    ----------
    W : sparse matrix or ndarray (data is float)
        Weight matrix. Mandatory.
    gtype : string
        Graph type (default is "unknown")
    lap_type : string
        Laplacian type (default is 'combinatorial')
    coords : ndarray
        Coordinates of the vertices (default is None)
    plotting : dict
        Dictionnary containing the plotting parameters

    Fields
    ------
    A graph contains the following fields:

    N : the number of nodes (also called vertices sometimes) in the graph.
        They represent the different points between which connections may occur.
    Ne : the number of edges (also called links sometimes) in the graph.
        They reprensent the actual connections between the nodes.
    W : the weight matrix contains the weights of the connections.
        It is represented as a NxN matrix of floats.
        W_i,j = 0 means that there is no connection from i to j.
    A : the adjacency matrix defines which edges exist on the graph.
        It is represented as a NxN matrix of booleans. A_i,j is True if W_i,j > 0.
    d : the degree vector of the vertices. It is represented as a Nx1 vector
        counting the number of connections that each node possesses.
    gtype : the graph type is a short description of the graph object.
    directed : the flag to assess if the graph is directed or not.
        In this framework, we consider that a graph is directed
        if and only if its weight matrix is non symmetric.
    L : the laplacian matrix. It is represented as a NxN matrix computed from W.
    lap_type : the laplacian type determine which kind of laplacian to compute.
        From a given matrix W, there exist several laplacians that could be computed.
    coords : the coordinates of the vertices in the 2D or 3D space for plotting.
    plotting : all the plotting parameters go here.
        They depend on the library used for plotting.


    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> W = np.arange(4).reshape(2, 2)
    >>> G = graphs.Graph(W)

    """
    def __init__(self, W, gtype='unknown', lap_type='combinatorial',
                 coords=None, plotting={}, **kwargs):

        self.logger = build_logger(__name__, **kwargs)

        shapes = np.shape(W)
        if len(shapes) != 2 or shapes[0] != shapes[1]:
            self.logger.error('W has incorrect shape {}'.format(shapes))

        self.N = shapes[0]
        self.W = sparse.lil_matrix(W)
        self.A = self.W > 0

        self.Ne = self.W.nnz
        self.d = self.A.sum(axis=1)
        self.gtype = gtype
        self.lap_type = lap_type

        self.is_directed()
        self.create_laplacian(lap_type)

        if isinstance(coords, np.ndarray) and 2 <= len(np.shape(coords)) <= 3:
            self.coords = coords
        else:
            self.coords = np.ndarray(None)

        # Plotting default parameters
        self.plotting = {'vertex_size': plotting.get('vertex_size', 10),
                         'edge_width': plotting.get('edge_width', 1)
                         'edge_style': plotting.get('edge_style', '-')
                         'vertex_color': plotting.get('vertex_color', 'b')
                         'edge_color': plotting.get('edge_color', np.array([255, 88, 41])/255.)}

        if isinstance(plotting, dict):
            self.plotting.update(plotting)

    def update_graph_attr(self, *args, **kwargs):
        r"""
        Recompute some attribute of the graph.

        Parameters
        ----------
        args: list of string
            the arguments that will be not changed and not re-compute.
        kwargs: Dictionnary
            The arguments with their new value.

        Return
        ------
        The same Graph with some updated values.

        Note
        ----
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

        Updates all attributes of G excepted 'N' and 'd'

        """
        graph_attr = {}
        valid_attributes = ['W', 'A', 'N', 'd', 'Ne', 'gtype', 'directed',
                            'coords', 'lap_type', 'L', 'plotting']

        for i in args:
            if i in valid_attributes:
                graph_attr[i] = getattr(self, i)
            else:
                self.logger.warning('Your attribute {} do not figure is the valid_attributes who are {}'.format(i, valid_attributes))

        for i in kwargs:
            if i in valid_attributes:
                if i in graph_attr:
                    self.logger.info('You already give this attribute in the args. Therefore, it will not be recaculate.')
                else:
                    graph_attr[i] = kwargs[i]
            else:
                self.logger.warning('Your attribute {} do not figure is the valid_attributes who are {}'.format(i, valid_attributes))

        from nngraphs import NNGraph
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

    def set_coords(self, kind='ring2D', coords=None):
        r"""
        Set coordinates for the vertices.

        Parameters
        ----------
        kind : string
            The kind of display. Default is 'ring2D'.
            Accepting ['ring2D', 'community2D', 'manual', 'random2D', 'random3D'].
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
        if kind not in ['ring2D', 'community2D', 'manual', 'random2D', 'random3D']:
            raise ValueError('Unexpected kind argument. Got {}.'.format(kind))

        if kind == 'manual':
            if isinstance(coords, list):
                coords = np.array(coords)
            if isinstance(coords, np.ndarray) and len(coords.shape) == 2 and \
                    coords.shape[0] == self.N and 2 <= coords.shape[1] <= 3:
                self.coords = coords
            else:
                raise ValueError('Expecting coords to be a list or ndarray of size Nx2 or Nx3.')

        elif kind == 'ring2D':
            tmp = np.arange(self.N).reshape(self.N, 1)
            self.coords = np.concatenate((np.cos(tmp*2*np.pi/self.N),
                                          np.sin(tmp*2*np.pi/self.N)),
                                         axis=1)

        elif kind == 'random2D':
            self.coords = np.random.rand((self.N, 2))

        elif kind == 'random3D':
            self.coords = np.random.rand((self.N, 3))

        elif kind == 'community2D':
            if not hasattr(self, 'info') or 'node_com' not in self.info:
                ValueError('Missing arguments to the graph to be able to compute community coordinates.')

            if 'world_rad' not in self.info:
                self.info['world_rad'] = np.sqrt(self.N)

            if 'comm_sizes' not in self.info:
                counts = Counter(self.info['node_com'])
                self.info['comm_sizes'] = np.array(list(map(lambda counter: counter[1], sorted(counts.items()))))

            Nc = self.info['comm_sizes'].shape[0]

            self.info['com_coords'] = self.info['world_rad'] * np.array(list(zip(
                np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc),
                np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))

            coords = np.random.rand(N, 2)  # nodes' coordinates inside the community
            self.coords = np.array([[elem[0] * np.cos(2 * np.pi * elem[1]),
                                elem[0] * np.sin(2 * np.pi * elem[1])] for elem in coords])

            for i in range(N):
                # set coordinates as an offset from the center of the community it belongs to
                comm_idx = self.info['node_com'][i]
                comm_rad = np.sqrt(self.info['comm_sizes'][comm_idx])
                self.coords[i] = self.info['com_coords'][comm_idx] + comm_rad * self.coords[i]

    def subgraph(self, ind):
        r"""
        Create a subgraph from G.

        Parameters
        ----------
        G : Graph
            Original graph
        ind : list
            Nodes to keep

        Returns
        -------
        subG : Graph
            Subgraph

        Examples
        --------
        >>> from pygsp import graphs
        >>> import numpy as np
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> ind = [3]
        >>> subG = graphs.Graph.subgraph(G, ind)

        This function create a subgraph from G taking only the nodes in ind.

        """
        if not isinstance(ind, list) and not isinstance(ind, np.ndarray):
            raise TypeError('The indices must be a list or a ndarray.')

        sub_G = self
        sub_G.W = self.W[np.ix_(ind, ind)]

        print self.W

        try:
            sub_G.N = len(ind)
        except TypeError:
            sub_G.N = 1

        sub_G.gtype = "sub-" + self.gtype

        return sub_G

    def is_connected(self):
        r"""
        Function to check the strong connectivity of the input graph.

        It uses DFS travelling on graph to ensure that each node is visited.
        For undirected graphs, starting at any vertex and trying to access all others is enough.
        For directed graphs, one needs to check that a random vertex is accessible by all others
        and can access all others. Thus, we can transpose the adjacency matrix and compute again
        with the same starting point in both phases.

        Returns
        -------
        is_connected : bool
            A bool value telling if the graph is connected

        Examples
        --------
        >>> from scipy import sparse
        >>> from pygsp import graphs
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> is_connected = G.is_connected()<

        """
        if not hasattr(self, 'directed'):
            self.is_directed()

        for adj_matrix in [self.A, self.A.T] if self.directed else [self.A]:
            visited = np.zeros(self.N, dtype=bool)
            stack = [0]

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    # Add indices of nodes not visited yet and accessible from v
                    stack.extend(filter(lambda idx: not visited[idx] and idx not in stack, adj_matrix[v, :].nonzero()[1]))

            if not visited.all():
                self.is_connected = False
                return False

        self.is_connected = True
        return True

    def is_directed(self):
        r"""
        Define if the graph has directed edges.

        Notes
        -----
        Can also be used to check if a matrix is symmetrical

        Examples
        --------
        >>> from scipy import sparse
        >>> from pygsp import graphs
        >>> W = sparse.rand(10, 10, 0.2)
        >>> G = graphs.Graph(W=W)
        >>> is_directed = G.is_directed()

        """
        if np.diff(np.shape(self.W))[0]:
            raise ValueError("Matrix dimensions mismatch, expecting square matrix.")

        is_dir = np.abs(self.W - self.W.T).sum() != 0

        self.directed = is_dir

    def compute_fourier_basis(self, smallest_first=True, force_recompute=False, **kwargs):
        r"""
        Compute the fourier basis of the graph.

        Parameters
        ----------
        smallest_first: bool
            Define the order of the eigenvalues.
            Default is smallest first (True).
        force_recompute: bool
            Force to recompute the Fourier basis if already existing.

        Note
        ----
        'G.compute_fourier_basis()' computes a full eigendecomposition of
        the graph Laplacian G.L:

        .. L = U Lambda U*

        .. math:: {\cal L} = U \Lambda U^*

        where $\Lambda$ is a diagonal matrix of the Laplacian eigenvalues.

        *G.e* is a column vector of length *G.N* containing the Laplacian
        eigenvalues. The largest eigenvalue is stored in *G.lmax*.
        The eigenvectors are stored as column vectors of *G.U* in the same
        order that the eigenvalues. Finally, the coherence of the
        Fourier basis is in *G.mu*.

        Example
        -------
        >>> from pygsp import graphs
        >>> N = 50
        >>> G = graphs.Sensor(N)
        >>> G.compute_fourier_basis()

        References
        ----------
        cite ´chung1997spectral´

        """
        if hasattr(self, 'e') or hasattr(self, 'U'):
            if force_recompute:
                logger.warning("This graph already has a Fourier basis. Recomputing.")
            else:
                logger.error("This graph already has a Fourier basis. Stopping.")
                return

        if self.N > 3000:
            logger.warning("Performing full eigendecomposition of a large "
                           "matrix may take some time.")

        if not hasattr(self, 'L'):
            raise AttributeError("Graph Laplacian is missing")

        eigenvectors, eigenvalues, _ = sp.linalg.svd(self.L.todense())

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
            The laplacian type to use. Default is "combinatorial".

        """
        if sp.shape(self.W) == (1, 1):
            self.L = sparse.lil_matrix(0)
            return

        if lap_type in ['combinatorial', 'normalized', 'none']:
            self.lap_type = lap_type
        else:
            raise AttributeError('Unknown laplacian type!')

        if self.directed:
            if lap_type == 'combinatorial':
                L = 0.5*(sparse.diags(np.ravel(self.W.sum(0)), 0) + sparse.diags(np.ravel(self.W.sum(1)), 0) - self.W - self.W.T).tocsc()
            elif lap_type == 'normalized':
                raise NotImplementedError('Yet. Ask Nathanael.')
            elif lap_type == 'none':
                L = sparse.lil_matrix(0)

        else:
            if lap_type == 'combinatorial':
                L = (sparse.diags(np.ravel(self.W.sum(1)), 0) - self.W).tocsc()
            elif lap_type == 'normalized':
                D = sparse.diags(np.ravel(np.power(self.W.sum(1), -0.5)), 0).tocsc()
                L = sparse.identity(self.N) - D * self.W * D
            elif lap_type == 'none':
                L = sparse.lil_matrix(0)

        self.L = L

    def estimate_lmax(self, force_recompute):
        r"""
        Estimate the maximal eigenvalue.

        Examples
        --------
        Just define a graph and apply the estimation on it.

        >>> from pygsp import graphs
        >>> import numpy as np
        >>> W = np.arange(16).reshape(4, 4)
        >>> G = graphs.Graph(W)
        >>> G.estimate_lmax()

        """
        if hashattr(self, 'lmax'):
            if force_recompute:
                logger.error('Already computed lmax. Recomputing.')
            else:
                logger.error('Already computed lmax. Stopping.')
                return

        try:
            lmax = sparse.linalg.eigs(self.L, k=1, tol=5e-3, ncv=10)[0]

            # On robustness purposes, increasing the error by 1 percent
            lmax *= 1.01
        except ValueError:
            logger.warning('GSP_ESTIMATE_LMAX: Cannot use default method.')
            lmax = np.max(self.d)

        self.lmax = np.real(lmax)

    def plot(self, **kwargs):
        r"""
        Plot the graph.

        See plotting doc.
        """
        from pygsp import plotting
        plotting.plot_graph(self, **kwargs)
