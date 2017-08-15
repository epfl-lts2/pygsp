# -*- coding: utf-8 -*-

from ..utils import build_logger
from ..data_handling import adj2vec

import numpy as np
from scipy import sparse


logger = build_logger(__name__)


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


def grad_mat(G):  # 1 call (above)
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
        adj2vec(G)

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


def gft(G, f):
    r"""
    Compute Graph Fourier transform.

    Parameters
    ----------
    G : Graph or Fourier basis
    f : ndarray
        must be in 2d, even if the second dim is 1 signal

    Returns
    -------
    f_hat : ndarray
        Graph Fourier transform of *f*
    """

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if not hasattr(G, 'U'):
            logger.info('Analysis filter has to compute the eigenvalues ' +
                        'and the eigenvectors.')
            G.compute_fourier_basis()

        U = G.U
    else:
        U = G

    return np.dot(np.conjugate(U.T), f)  # True Hermitian here.


def igft(G, f_hat):
    r"""
    Compute inverse graph Fourier transform.

    Parameters
    ----------
    G : Graph or Fourier basis
    f_hat : ndarray
        Signal

    Returns
    -------
    f : ndarray
        Inverse graph Fourier transform of *f_hat*

    """

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if not hasattr(G, 'U'):
            logger.info('Analysis filter has to compute the eigenvalues ' +
                        'and the eigenvectors.')
            G.compute_fourier_basis()
        U = G.U

    else:
        U = G

    return np.dot(U, f_hat)


def localize(g, i):
    r"""
    Localize a kernel g to the node i.

    Parameters
    ----------
    g : Filter
        kernel (or filterbank)
    i : int
        Index of vertex

    Returns
    -------
    gt : ndarray
        Translated signal

    """
    N = g.G.N
    f = np.zeros((N))
    f[i - 1] = 1

    gt = np.sqrt(N) * g.analysis(f)

    return gt


def modulate(G, f, k):
    r"""
    Modulation the signal f to the frequency k.

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal (column)
    k :  int
        Index of frequencies

    Returns
    -------
    fm : ndarray
        Modulated signal

    """
    nt = np.shape(f)[1]
    fm = np.sqrt(G.N) * np.kron(np.ones((nt, 1)), f) * \
        np.kron(np.ones((1, nt)), G.U[:, k])

    return fm


def translate(G, f, i):
    r"""
    Translate the signal f to the node i.

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal
    i : int
        Indices of vertex

    Returns
    -------
    ft : translate signal

    """

    fhat = gft(G, f)
    nt = np.shape(f)[1]

    ft = np.sqrt(G.N) * igft(G, fhat, np.kron(np.ones((1, nt)), G.U[i]))

    return ft
