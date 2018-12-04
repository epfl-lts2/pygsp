# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy import sparse

from pygsp import utils


logger = utils.build_logger(__name__)


class GraphDifference(object):

    @property
    def D(self):
        r"""Differential operator (for gradient and divergence).

        Is computed by :func:`compute_differential_operator`.
        """
        if not hasattr(self, '_D'):
            self.logger.warning('The differential operator G.D is not '
                                'available, we need to compute it. Explicitly '
                                'call G.compute_differential_operator() '
                                'once beforehand to suppress the warning.')
            self.compute_differential_operator()
        return self._D

    def compute_differential_operator(self):
        r"""Compute the graph differential operator (cached).

        The differential operator is the matrix :math:`D` such that

        .. math:: L = D D^\top,

        where :math:`L` is the graph Laplacian (combinatorial or normalized).
        It is used to compute the gradient and the divergence of graph
        signals (see :meth:`grad` and :meth:`div`).

        The differential operator computes the gradient and divergence of
        signals, and the Laplacian computes the divergence of the gradient, as
        follows:

        .. math:: z = L x = D y = D D^\top x,

        where :math:`y = D^\top x = \nabla_\mathcal{G} x` is the gradient of
        :math:`x` and :math:`z = D y = \operatorname{div}_\mathcal{G} y` is the
        divergence of :math:`y`. See :meth:`grad` and :meth:`div` for details.

        The difference operator is actually an incidence matrix of the graph,
        defined as

        .. math:: D[i, k] =
            \begin{cases}
                -\sqrt{W[i, j] / 2} &
                    \text{if } e_k = (v_i, v_j) \text{ for some } j, \\
                +\sqrt{W[i, j] / 2} &
                    \text{if } e_k = (v_j, v_i) \text{ for some } j, \\
                0 & \text{otherwise}
            \end{cases}

        for the combinatorial Laplacian, and

        .. math:: D[i, k] =
            \begin{cases}
                -\sqrt{W[i, j] / 2 / d[i]} &
                    \text{if } e_k = (v_i, v_j) \text{ for some } j, \\
                +\sqrt{W[i, j] / 2 / d[i]} &
                    \text{if } e_k = (v_j, v_i) \text{ for some } j, \\
                0 & \text{otherwise}
            \end{cases}

        for the normalized Laplacian, where :math:`v_i \in \mathcal{V}` is a
        vertex, :math:`e_k = (v_i, v_j) \in \mathcal{E}` is an edge from
        :math:`v_i` to :math:`v_j`, :math:`W[i, j]` is the weight :attr:`W` of
        the edge :math:`(v_i, v_j)`, :math:`d[i]` is the degree :attr:`dw` of
        vertex :math:`v_i`.

        For undirected graphs, only half the edges are kept (the upper
        triangular part of the adjacency matrix) in the interest of space and
        time. In that case, the :math:`1/\sqrt{2}` factor disappears from the
        above equations for :math:`L = D D^\top` to stand at all times.

        The result is cached and accessible by the :attr:`D` property.

        See also
        --------
        grad : compute the gradient
        div : compute the divergence

        Examples
        --------

        The difference operator is an incidence matrix.
        Example with a undirected graph.

        >>> adjacency = np.array([
        ...     [0, 2, 0],
        ...     [2, 0, 1],
        ...     [0, 1, 0],
        ... ])
        >>> graph = graphs.Graph(adjacency)
        >>> graph.compute_laplacian('combinatorial')
        >>> graph.compute_differential_operator()
        >>> graph.D.toarray()
        array([[-1.41421356,  0.        ],
               [ 1.41421356, -1.        ],
               [ 0.        ,  1.        ]])
        >>> graph.compute_laplacian('normalized')
        >>> graph.compute_differential_operator()
        >>> graph.D.toarray()
        array([[-1.        ,  0.        ],
               [ 0.81649658, -0.57735027],
               [ 0.        ,  1.        ]])

        Example with a directed graph.

        >>> adjacency = np.array([
        ...     [0, 2, 0],
        ...     [2, 0, 1],
        ...     [0, 0, 0],
        ... ])
        >>> graph = graphs.Graph(adjacency)
        >>> graph.compute_laplacian('combinatorial')
        >>> graph.compute_differential_operator()
        >>> graph.D.toarray()
        array([[-1.        ,  1.        ,  0.        ],
               [ 1.        , -1.        , -0.70710678],
               [ 0.        ,  0.        ,  0.70710678]])
        >>> graph.compute_laplacian('normalized')
        >>> graph.compute_differential_operator()
        >>> graph.D.toarray()
        array([[-0.70710678,  0.70710678,  0.        ],
               [ 0.63245553, -0.63245553, -0.4472136 ],
               [ 0.        ,  0.        ,  1.        ]])

        The graph Laplacian acts on a signal as the divergence of the gradient.

        >>> G = graphs.Logo()
        >>> G.compute_differential_operator()
        >>> s = np.random.normal(size=G.N)
        >>> s_grad = G.D.T.dot(s)
        >>> s_lap = G.D.dot(s_grad)
        >>> np.linalg.norm(s_lap - G.L.dot(s)) < 1e-10
        True

        """

        sources, targets, weights = self.get_edge_list()

        n = self.n_edges
        rows = np.concatenate([sources, targets])
        columns = np.concatenate([np.arange(n), np.arange(n)])
        values = np.empty(2*n)

        if self.lap_type == 'combinatorial':
            values[:n] = -np.sqrt(weights)
            values[n:] = -values[:n]
        elif self.lap_type == 'normalized':
            values[:n] = -np.sqrt(weights / self.dw[sources])
            values[n:] = +np.sqrt(weights / self.dw[targets])
        else:
            raise ValueError('Unknown lap_type {}'.format(self.lap_type))

        if self.is_directed():
            values /= np.sqrt(2)

        self._D = sparse.csc_matrix((values, (rows, columns)),
                                    shape=(self.n_vertices, self.n_edges))
        self.D.eliminate_zeros()  # Self-loops introduce stored zeros.

    def grad(self, s):
        r"""Compute the gradient of a signal defined on the vertices.

        The gradient of a signal :math:`s` is defined as

        .. math:: y = D^\top s,

        where :math:`D` is the differential operator :attr:`D`.

        Parameters
        ----------
        s : ndarray
            Signal of length :attr:`n_vertices` living on the vertices.

        Returns
        -------
        s_grad : ndarray
            Gradient signal of length :attr:`n_edges` living on the edges.

        See also
        --------
        compute_differential_operator
        div : compute the divergence of an edge signal

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 3131)
        >>> s = np.random.normal(size=G.N)
        >>> s_grad = G.grad(s)
        >>> s_div = G.div(s_grad)
        >>> np.linalg.norm(s_div - G.L.dot(s)) < 1e-10
        True

        """
        if self.N != s.shape[0]:
            raise ValueError('Signal length should be the number of nodes.')
        return self.D.T.dot(s)

    def div(self, s):
        r"""Compute the divergence of a signal defined on the edges.

        The divergence of a signal :math:`s` is defined as

        .. math:: y = D s,

        where :math:`D` is the differential operator :attr:`D`.

        Parameters
        ----------
        s : ndarray
            Signal of length :attr:`n_edges` living on the edges.

        Returns
        -------
        s_div : ndarray
            Divergence signal of length :attr:`n_vertices` living on the
            vertices.

        See also
        --------
        compute_differential_operator
        grad : compute the gradient of a node signal

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 3131)
        >>> s = np.random.normal(size=G.Ne)
        >>> s_div = G.div(s)
        >>> s_grad = G.grad(s_div)

        """
        if self.Ne != s.shape[0]:
            raise ValueError('Signal length should be the number of edges.')
        return self.D.dot(s)
