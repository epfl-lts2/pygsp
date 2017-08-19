# -*- coding: utf-8 -*-

from pygsp import utils


logger = utils.build_logger(__name__)


def div(G, s):
    r"""
    Compute the graph divergence of a signal.

    Parameters
    ----------
    G : Graph
    s : ndarray
        Signal of length G.Ne/2 living on the edges (non-directed graph).

    Returns
    -------
    divergence : ndarray
        Divergence signal of length G.N living on the nodes.

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, operators
    >>> G = graphs.Logo()
    >>> G.N, G.Ne
    (1130, 6262)
    >>> s = np.random.normal(size=G.Ne//2)  # Symmetric weight matrix.
    >>> div = operators.div(G, s)
    >>> grad = operators.grad(G, div)

    """
    if G.Ne != 2 * s.shape[0]:
        raise ValueError('Signal length should be the number of edges.')
    return G.D.T.dot(s)


def grad(G, s):
    r"""
    Compute the graph gradient of a signal.

    Parameters
    ----------
    G : Graph
    s : ndarray
        Signal of length G.N living on the nodes.

    Returns
    -------
    gradient : ndarray
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
    >>> grad = operators.grad(G, s)
    >>> div = operators.div(G, grad)

    """
    if G.N != s.shape[0]:
        raise ValueError('Signal length should be the number of nodes.')
    return G.D.dot(s)
