# -*- coding: utf-8 -*-

from pygsp import utils


logger = utils.build_logger(__name__)


class GraphDifference(object):

    def grad(self, s):
        r"""
        Compute the graph gradient of a signal.

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
        >>> import numpy as np
        >>> from pygsp import graphs
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 6262)
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
        r"""
        Compute the graph divergence of a signal.

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
        >>> import numpy as np
        >>> from pygsp import graphs
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 6262)
        >>> s = np.random.normal(size=G.Ne//2)  # Symmetric weight matrix.
        >>> s_div = G.div(s)
        >>> s_grad = G.grad(s_div)

        """
        if self.Ne != 2 * s.shape[0]:
            raise ValueError('Signal length should be the number of edges.')
        return self.D.T.dot(s)
