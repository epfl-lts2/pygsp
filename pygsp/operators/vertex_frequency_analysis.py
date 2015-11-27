# -*- coding: utf-8 -*-

from pygsp import data_handling
from pygsp.operators import operator
from pygsp.utils import build_logger

import numpy as np

logger = build_logger(__name__)


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

    if not hasattr(G, 'U'):
        logger.info('Analysis filter has to compute the eigenvalues and the eigenvectors.')
        G.compute_fourier_basis()

    if isinstance(g, list):
        g = operator.igft(G, g[0](G.e))
    elif hasattr(g, '__call__'):
        g = operator.igft(G, g(G.e))

    if not lowmemory:
        # Compute the Frame into a big matrix
        Frame = gwft_frame_matrix(G, g)

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
    k : #TODO
        kernel

    Returns
    -------
    C : Coefficient.
    """
    from pygsp.filters import Gabor

    if not hasattr(G, 'e'):
        logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
        G.compute_fourier_basis()
    g = Gabor(G, k)

    C = g.analysis(f)
    C = data_handling.vec2mat(C, G.N).T

    return C


def gwft_frame_matrix(G, g):
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
    F = data_handling.repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

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

    if not hasattr(G, 'U'):
        logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
        G.compute_fourier_basis()

    if lowmemory:
        # Compute the Frame into a big matrix
        Frame = ngwft_frame_matrix(G, g)
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


def ngwft_frame_matrix(G, g):
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

    F = data_handling.repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    # Normalization
    F /= np.kron((np.ones((G.N)), np.sqrt(np.sum(np.power(np.abs(F), 2),
                                          axiis=0))))

    return F
