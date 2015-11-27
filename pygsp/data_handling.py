# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse


def adj2vec(G):
    r"""
    Prepare the graph for the gradient computation.

    Parameters
    ----------
    G : Graph structure

    """
    if not hasattr(G, 'directed'):
        G.is_directed()

    if G.directed:
        raise NotImplementedError("Not implemented yet.")

    else:
        v_i, v_j = (sparse.tril(G.W)).nonzero()
        weights = G.W[v_i, v_j]

        # TODO G.ind_edges = sub2ind(size(G.W), G.v_in, G.v_out)
        G.v_in = v_i
        G.v_out = v_j
        G.weights = weights
        G.Ne = np.shape(v_i)[0]

    # TODO Return vec


def mat2vec(d):
    r"""Not implemented yet"""
    raise NotImplementedError


def repmatline(A, ncol=1, nrow=1):
    r"""
    Repeat the matrix A in a specific manner.

    Parameters
    ----------
    A : ndarray
    ncol : Integer
        default is 1
    nrow : Integer
        default is 1

    Returns
    -------
    Ar : Matrix

    Examples
    --------

    For nrow=2 and ncol=3, the matrix
    ::

        x   =   [1 2 ]
                [3 4 ]

    becomes
    ::

                [1 1 1 2 2 2 ]
        M   =   [1 1 1 2 2 2 ]
                [3 3 3 4 4 4 ]
                [3 3 3 4 4 4 ]

    with::
        M = np.repeat(np.repeat(x, nrow, axis=1), ncol, axis=0)

    """
    if ncol < 1 or nrow < 1:
        raise ValueError("The number of lines and rows must be greater or\
                         equal to one, or you will get an empty array.")

    return np.repeat(np.repeat(A, ncol, axis=1), nrow, axis=0)


def vec2mat(d, Nf):
    r"""
    Vector to matrix transformation.

    Parameters
    ----------
    d : Ndarray
        Data
    Nf : int
        Number of filter

    Returns
    -------
    d : list of ndarray
        Data

    """
    if len(np.shape(d)) == 1:
        M = np.shape(d)[0]
        return np.reshape(d, (M / Nf, Nf), order='F')

    if len(np.shape(d)) == 2:
        M, N = np.shape(d)
        return np.reshape(d, (M / Nf, Nf, N), order='F')
