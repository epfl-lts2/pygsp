# -*- coding: utf-8 -*-

from pygsp import utils
from pygsp.graphs.gutils import compute_fourier_basis
from pygsp.data_handling import adj2vec

import numpy as np
from scipy import sparse
from math import sqrt


logger = utils.build_logger(__name__)


def div(G, s):
    r"""
    Parameters
    ----------
    G : Graph structure
    s : ndarray
        Signal living on the nodes

    Returns
    -------
    """
    if hasattr(G, 'lap_type'):
        if G.lap_type == 'combinatorial':
            raise NotImplementedError('Not implemented yet. However ask Nathanael it is very easy')

    if G.Ne != np.shape(s)[0]:
        raise ValueError('Signal size not equal to number of edges')

    D = grad_mat(G)
    di = D.getH() * s

    if s.dtype == 'float32':
        di = np.float32(di)

    return di


def grad(G, s):
    r"""
    Graph gradient
    Usage: gr = gsp_grad(G,s)

    Parameters
    ----------
    G : Graph structure
    s : ndarray
        Signal living on the nodes

    Returns
    -------
    gr : Gradient living on the edges

    """
    if hasattr(G, 'lap_type'):
        if G.lap_type == 'combinatorial':
            raise NotImplementedError('Not implemented yet. However ask Nathanael it is very easy')

    D = grad_mat(G)
    gr = D * s

    if s.dtype == 'float32':
        gr = np.float32(gr)

    return gr


def grad_mat(G):
    r"""
    Gradient sparse matrix of the graph G
    Usage:  D = gsp_gradient_mat(G);

    Parameters
    ----------
    G : Graph structure

    Returns
    -------
    D : Gradient sparse matrix

    """
    if not hasattr(G, 'v_in'):
        G = adj2vec(G)
        logger.info('To be more efficient you should run: G = adj2vec(G); \
              before using this proximal operator.')

    if hasattr(G, 'Diff'):
        D = G.Diff

    else:
        n = G.Ne
        Dc = np.ones((2 * n))
        Dv = np.ones((2 * n))

        Dr = np.concatenate((np.arange(n), np.arange(n)))
        Dc[:n] = G.v_in
        Dc[n:] = G.v_out
        Dv[:n] = np.sqrt(G.weights)
        Dv[n:] = -np.sqrt(G.weight)
        D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, G.N))

    return D


def gft(G, f):
    r"""
    Graph Fourier transform

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
            logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
            compute_fourier_basis(G)

        U = G.U
    else:
        U = G

    return np.dot(np.conjugate(U.T), f)


def igft(G, f_hat):
    r"""
    Inverse graph Fourier transform

    Parameters
    ----------
    G : Graph or Fourier basis
    f_hat : ndarray
        Signal

    Returns
    -------
    f : Inverse graph Fourier transform of *f_hat*

    """

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if not hasattr(G, 'U'):
            logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
            compute_fourier_basis(G)
        U = G.U

    else:
        U = G

    return np.dot(U, f_hat)


def localize(G, g, i):
    r"""
    Localize a kernel g to the node i

    Parameters
    ----------
    G : Graph
    g : Filter
        kernel (or filterbank)
    i : int
        Indices of vertex

    Returns
    -------
    gt : translate signal

    """
    f = np.zeros((G.N))
    f[i - 1] = 1

    gt = sqrt(G.N) * g.analysis(G, f)

    return gt


def modulate(G, f, k):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal (column)
    k :  int
        Indices of frequencies

    Returns
    -------
    fm : Modulated signal

    """
    nt = np.shape(f)[1]
    fm = np.sqrt(G.N)*np.kron(np.ones((nt, 1)), f)*np.kron(np.ones((1, nt)), G.U[:, k])

    return fm


def translate(G, f, i):
    r"""
    Tranlate the signal f to the node i

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

    ft = np.sqrt(G.N)*igft(G, fhat, np.kron(np.ones((1, nt)), G.U[i]))

    return ft
