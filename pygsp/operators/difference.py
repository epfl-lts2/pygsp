# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils


logger = utils.build_logger(__name__)


def div(G, s):
    r"""
    Compute Graph divergence of a signal.

    Parameters
    ----------
    G : Graph structure
    s : ndarray
        Signal living on the nodes

    Returns
    -------
    di : float
        The graph divergence

    """
    if G.Ne != np.shape(s)[0]:
        raise ValueError('Signal size is different from the number of edges.')

    D = grad_mat(G)
    di = D.T * s

    if s.dtype == 'float32':
        di = np.float32(di)

    return di


def grad(G, s):
    r"""
    Compute the Graph gradient.

    Examples
    --------
    >>> import pygsp
    >>> import numpy as np
    >>> G = pygsp.graphs.Logo()
    >>> s = np.random.rand(G.N)
    >>> grad = pygsp.operators.grad(G, s)

    Parameters
    ----------
    G : Graph structure
    s : ndarray
        Signal living on the nodes

    Returns
    -------
    gr : ndarray
        Gradient living on the edges

    """
    if G.N != np.shape(s)[0]:
        raise ValueError('Signal size is different from the number of nodes.')

    D = grad_mat(G)
    gr = D * s

    if s.dtype == 'float32':
        gr = np.float32(gr)

    return gr


def grad_mat(G):
    r"""
    Gradient sparse matrix of the graph G.

    Examples
    --------
    >>> import pygsp
    >>> G = pygsp.graphs.Logo()
    >>> D = pygsp.operators.grad_mat(G)

    Parameters
    ----------
    G : Graph structure

    Returns
    -------
    D : ndarray
        Gradient sparse matrix

    """
    if not hasattr(G, 'v_in'):
        utils.adj2vec(G)

    if hasattr(G, 'Diff'):
        if not sparse.issparse(G.Diff):
            G.Diff = sparse.csc_matrix(G.Diff)
        D = G.Diff

    else:
        n = G.Ne
        Dr = np.concatenate((np.arange(n), np.arange(n)))
        Dc = np.ones((2 * n))
        Dc[:n] = G.v_in
        Dc[n:] = G.v_out
        Dv = np.ones((2 * n))

        try:
            if G.lap_type == 'combinatorial':
                Dv[:n] = np.sqrt(G.weights.toarray())
                Dv[n:] = -Dv[:n]

            elif G.lap_type == 'normalized':
                Dv[:n] = np.sqrt(G.weights.toarray() / G.d[G.v_in])
                Dv[n:] = -np.sqrt(G.weights.toarray() / G.d[G.v_out])

            else:
                raise NotImplementedError('grad not implemented yet for ' +
                                          'this type of graph Laplacian.')
        except AttributeError as err:
            print('Graph does not have lap_type attribute: ' + str(err))

        D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, G.N))
        G.Diff = D

    return D
