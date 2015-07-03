# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse


def adj2vec(G):
    r"""
    Prepare the graph for the gradient computation

    Parameters
    ----------
    G : Graph structure
    """
    from operators.operator import grad_mat

    if G.directed:
        raise NotImplementedError("Not implemented yet")

    else:
        v_i, v_j = (sparse.tril(G.W)).nonzero()
        weights = G.W[v_i, v_j]

        # TODO G.ind_edges = sub2ind(size(G.W), G.v_in, G.v_out)
        G.v_in = v_i
        G.v_out = v_j
        G.weights = weights
        G.Ne = np.shape(v_i)[0]

        G.Diff = grad_mat(G)


def mat2vec(d):
    raise NotImplementedError


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid

    Parameters
    ----------
    ca : ndarray
        Array with the coarse approximation at each level
    pe : ndarray
        Array with the prediction errors at each level

    Returns
    -------
    coeff : ndarray
        Array of coefficient
    """
    Nl = len(ca) - 1
    N = 0

    for ele in ca:
        N += np.shape(ele)[0]

    try:
        Nt, Nv = np.shape(ca[Nl])
        coeff = coeff = np.zeros((N, Nv))
    except ValueError:
        Nt = np.shape(ca[Nl])[0]
        coeff = np.zeros((N))

    coeff[:Nt] = ca[Nl]

    ind = Nt
    for i in range(Nl):
        Nt = np.shape(ca[Nl - 1 - i])[0]
        coeff[ind + np.arange(Nt)] = pe[Nl - 1 - i]
        ind += Nt

    if ind != N:
        raise ValueError('Something is wrong here: contact the gspbox team.')

    return coeff


def repmatline(A, ncol=1, nrow=1):
    r"""
    This function repeats the matrix A in a specific manner.

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

    For ncol=2 and nrow=3, the matix

                1 2
                3 4
    becomes
                1 1 1 2 2 2
                1 1 1 2 2 2
                3 3 3 4 4 4
                3 3 3 4 4 4
    np.repeat(np.repeat(x, nrow, axis=1), ncol,  axis=0)

    """

    if ncol < 0 or nrow < 0:
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
        return np.reshape(d, (M/Nf, Nf), order='F')

    if len(np.shape(d)) == 2:
        M, N = np.shape(d)
        return np.reshape(d, (M/Nf, Nf, N), order='F')
