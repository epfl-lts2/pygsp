# -*- coding: utf-8 -*-

from pygsp import utils


logger = utils.build_logger(__name__)


def div(G, s):
    r"""
    Compute the graph divergence of a signal.

    The divergence of a signal :math:`s` is defined as

    .. math:: y = D^T s,

    where :math:`D` is the differential operator
    :py:attr:`pygsp.graphs.Graph.D`.

    Parameters
    ----------
    G : Graph
    s : ndarray
        Signal of length G.Ne/2 living on the edges (non-directed graph).

    Returns
    -------
    s_div : ndarray
        Divergence signal of length G.N living on the nodes.

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, operators
    >>> G = graphs.Logo()
    >>> G.N, G.Ne
    (1130, 6262)
    >>> s = np.random.normal(size=G.Ne//2)  # Symmetric weight matrix.
    >>> s_div = operators.div(G, s)
    >>> s_grad = operators.grad(G, s_div)

    """
    if G.Ne != 2 * s.shape[0]:
        raise ValueError('Signal length should be the number of edges.')
    return G.D.T.dot(s)


def grad(G, s):
    r"""
    Compute the graph gradient of a signal.

    The gradient of a signal :math:`s` is defined as

    .. math:: y = D s,

    where :math:`D` is the differential operator
    :py:attr:`pygsp.graphs.Graph.D`.

    Parameters
    ----------
    G : Graph
    s : ndarray
        Signal of length G.N living on the nodes.

    Returns
    -------
    s_grad : ndarray
        Gradient signal of length G.Ne/2 living on the edges (non-directed
        graph).

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, operators
    >>> G = graphs.Logo()
    >>> G.N, G.Ne
    (1130, 6262)
    >>> s = np.random.normal(size=G.N)
    >>> s_grad = operators.grad(G, s)
    >>> s_div = operators.div(G, s_grad)
    >>> np.linalg.norm(s_div - G.L.dot(s)) < 1e-10
    True

    """
    if G.N != s.shape[0]:
        raise ValueError('Signal length should be the number of nodes.')
    return G.D.dot(s)
