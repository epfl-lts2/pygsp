# -*- coding: utf-8 -*-

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

        The differential operator is a matrix such that

        .. math:: L = D^T D,

        where :math:`D` is the differential operator and :math:`L` is the graph
        Laplacian. It is used to compute the gradient and the divergence of a
        graph signal, see :meth:`grad` and :meth:`div`.

        The result is cached and accessible by the :attr:`D` property.

        See also
        --------
        grad : compute the gradient
        div : compute the divergence

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 3131)
        >>> G.compute_differential_operator()
        >>> G.D.shape == (G.Ne, G.N)
        True

        """

        v_in, v_out, weights = self.get_edge_list()

        n = len(v_in)
        Dr = np.concatenate((np.arange(n), np.arange(n)))
        Dc = np.empty(2*n)
        Dc[:n] = v_in
        Dc[n:] = v_out
        Dv = np.empty(2*n)

        if self.lap_type == 'combinatorial':
            Dv[:n] = np.sqrt(weights)
            Dv[n:] = -Dv[:n]
        elif self.lap_type == 'normalized':
            Dv[:n] = np.sqrt(weights / self.dw[v_in])
            Dv[n:] = -np.sqrt(weights / self.dw[v_out])
        else:
            raise ValueError('Unknown lap_type {}'.format(self.lap_type))

        self._D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, self.N))

    def grad(self, s):
        r"""Compute the gradient of a graph signal.

        The gradient of a signal :math:`s` is defined as

        .. math:: y = D s,

        where :math:`D` is the differential operator :attr:`D`.

        Parameters
        ----------
        s : ndarray
            Signal of length G.N living on the nodes.

        Returns
        -------
        s_grad : ndarray
            Gradient signal of length G.Ne/2 living on the edges (non-directed
            graph).

        See also
        --------
        compute_differential_operator
        div : compute the divergence

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
        return self.D.dot(s)

    def div(self, s):
        r"""Compute the divergence of a graph signal.

        The divergence of a signal :math:`s` is defined as

        .. math:: y = D^T s,

        where :math:`D` is the differential operator :attr:`D`.

        Parameters
        ----------
        s : ndarray
            Signal of length G.Ne/2 living on the edges (non-directed graph).

        Returns
        -------
        s_div : ndarray
            Divergence signal of length G.N living on the nodes.

        See also
        --------
        compute_differential_operator
        grad : compute the gradient

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
        return self.D.T.dot(s)
