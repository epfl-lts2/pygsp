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

        The result is cached and accessible by the :attr:`D` property.

        See also
        --------
        grad : compute the gradient
        div : compute the divergence

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_differential_operator()
        >>> G.D.shape == (G.n_vertices, G.n_edges)
        True

        """

        sources, targets, weights = self.get_edge_list()

        n = self.n_edges
        rows = np.concatenate([sources, targets])
        columns = np.concatenate([np.arange(n), np.arange(n)])
        values = np.empty(2*n)

        if self.lap_type == 'combinatorial':
            values[:n] = np.sqrt(weights)
            values[n:] = -values[:n]
        elif self.lap_type == 'normalized':
            values[:n] = np.sqrt(weights / self.dw[sources])
            values[n:] = -np.sqrt(weights / self.dw[targets])
        else:
            raise ValueError('Unknown lap_type {}'.format(self.lap_type))

        self._D = sparse.csc_matrix((values, (rows, columns)),
                                    shape=(self.n_vertices, self.n_edges))

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
