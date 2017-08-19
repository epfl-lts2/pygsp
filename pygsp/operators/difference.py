# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

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

    D = grad_mat(G)
    return D.T * s


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

    D = grad_mat(G)
    return D * s


def grad_mat(G):
    r"""
    Compute the gradient sparse matrix of the graph G.

    Parameters
    ----------
    G : Graph

    Returns
    -------
    D : ndarray
        Gradient sparse matrix of size G.Ne/2 x G.N (non-directed graph).

    Examples
    --------
    >>> from pygsp import graphs, operators
    >>> G = graphs.Logo()
    >>> G.N, G.Ne
    (1130, 6262)
    >>> operators.grad_mat(G).shape == (G.Ne//2, G.N)
    True

    """
    if not hasattr(G, 'v_in'):
        utils.adj2vec(G)

    if hasattr(G, 'Diff'):
        if not sparse.issparse(G.Diff):
            G.Diff = sparse.csc_matrix(G.Diff)
        return G.Diff

    n = G.Ne // 2
    Dr = np.concatenate((np.arange(n), np.arange(n)))
    Dc = np.ones((2 * n))
    Dc[:n] = G.v_in
    Dc[n:] = G.v_out
    Dv = np.empty((2 * n))

    if not hasattr(G, 'lap_type'):
        raise ValueError('Graph does not have the lap_type attribute.')

    if G.lap_type == 'combinatorial':
        Dv[:n] = np.sqrt(G.weights.toarray())
        Dv[n:] = -Dv[:n]
    elif G.lap_type == 'normalized':
        Dv[:n] = np.sqrt(G.weights.toarray() / G.d[G.v_in])
        Dv[n:] = -np.sqrt(G.weights.toarray() / G.d[G.v_out])
    else:
        raise ValueError('Unknown lap_type {}'.format(G.lap_type))

    D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, G.N))
    G.Diff = D

    return G.Diff
