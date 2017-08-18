# -*- coding: utf-8 -*-

import numpy as np

from ..utils import build_logger
from ..data_handling import vec2mat, repmatline


logger = build_logger(__name__)


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
        U = G.U
    else:
        U = G

    return np.dot(U, f_hat)


def generalized_wft(G, g, f, lowmemory=True):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
        G : Graph
        g : ndarray or Filter
            Window (graph signal or kernel)
        f : ndarray
            Graph signal
        lowmemory : bool
            use less memory (default=True)

    Returns
    -------
        C : ndarray
            Coefficients

    """
    Nf = np.shape(f)[1]

    if isinstance(g, list):
        g = igft(G, g[0](G.e))
    elif hasattr(g, '__call__'):
        g = igft(G, g(G.e))

    if not lowmemory:
        # Compute the Frame into a big matrix
        Frame = _gwft_frame_matrix(G, g)

        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N, Nf), order='F')

    else:
        # Compute the translate of g
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N) * np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))
        C = np.zeros((G.N, G.N))

        for j in range(Nf):
            for i in range(G.N):
                C[:, i, j] = (np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.dot(np.kron(np.ones((G.N)), Ftrans[:, i])).T, f[:, j])

    return C


def gabor_wft(G, f, k):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
        G : Graph
        f : ndarray
            Graph signal
        k : anonymous function
            Gabor kernel

    Returns
    -------
        C : Coefficient.

    """
    from pygsp.filters import Gabor

    g = Gabor(G, k)

    C = g.analysis(f)
    C = vec2mat(C, G.N).T

    return C


def _gwft_frame_matrix(G, g):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
        G : Graph
        g : window

    Returns
    -------
        F : ndarray
            Frame
    """

    if G.N > 256:
        logger.warning("It will create a big matrix. You can use other methods.")

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))
    F = repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    return F


def ngwft(G, f, g, lowmemory=True):
    r"""
    Normalized graph windowed Fourier transform

    Parameters
    ----------
        G : Graph
        f : ndarray
            Graph signal
        g : ndarray
            Window
        lowmemory : bool
            Use less memory. (default = True)

    Returns
    -------
        C : ndarray
            Coefficients

    """

    if lowmemory:
        # Compute the Frame into a big matrix
        Frame = _ngwft_frame_matrix(G, g)
        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N), order='F')

    else:
        # Compute the translate of g
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))
        C = np.zeros((G.N, G.N))

        for i in range(G.N):
            atoms = np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.kron(np.ones((G.N)), Ftrans[:, i]).T

            # normalization
            atoms /= np.kron((np.ones((G.N))), np.sqrt(np.sum(np.abs(atoms),
                                                              axis=0)))
            C[:, i] = np.dot(atoms, f)

    return C


def _ngwft_frame_matrix(G, g):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
        G : Graph
        g : ndarray
            Window

    Output parameters:
        F : ndarray
            Frame
    """
    if G.N > 256:
        logger.warning('It will create a big matrix, you can use other methods.')

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(g.N)*np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))

    F = repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    # Normalization
    F /= np.kron((np.ones((G.N)), np.sqrt(np.sum(np.power(np.abs(F), 2),
                                          axiis=0))))

    return F
