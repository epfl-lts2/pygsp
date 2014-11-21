import numpy as np
from scipy import sparse
from math import isinf, isnan


def is_directed(G):
    r"""
    Returns a bool:  True if the graph is directed and false if not
    The Weight matrix has to be sparse (For now)
    Can also be used to check if a matrix is symetrical
    """

    is_dir = (G.W - G.W.sparse.transpose()).sum() != 0
    return is_dir


def estimate_lmax(G):
    r"""
    TODO write doc
    """
    try:
        # MAT: lmax=eigs(G.L,1,'lm',opts)
        lmax = sparse.linalg.eigs(G.L, k=1, tol=5e-3, ncv=10)[0]
        # On robustness purposes, increasing the error by 1 percent
        lmax *= 1.01
    except ValueError:
        print('GSP_ESTIMATE_LMAX: Cannot use default method')
        lmax = max(G.d)
    return lmax


def check_weights(W):
    r"""
    Check a weight matrix
    Returns an array of bools:
        has_inf_val
        has_nan_value
        is_not_square
        diag_is_not_zero
    """
    has_inf_val = False
    diag_is_not_zero = False
    is_not_square = False
    has_nan_value = False
    if isinf(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an inifinite \
              value in the weight matrix")
        has_inf_val = True
    if abs(W.diagonal()).sum():
        print("GSP_TEST_WEIGHTS: The main diagonal of \
              the weight matrix is not 0!")
        diag_is_not_zero = True
    if W.get_shape()[0] != W.get_shape()[1]:
        print("GSP_TEST_WEIGHTS: The weight matrix is \
              not square!")
        is_not_square = True
    if isnan(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an inifinite \
              value in the weight matrix")
        has_nan_value = True

    return [has_inf_val, has_nan_value, is_not_square, diag_is_not_zero]


def create_laplacian(G):
    # TODO ask
    # if size(G):
    #     Ng = size(G)
    #     i = 0
    #     # TODO check indice
    #     while i < Ng:
    #         G[i] = create_laplacian(G[i])
    if G.gtype == 'combinatorial':
        L = sparse.lil_matrix(G.W.sum().diagonal() - G.W)
    if G.gtype == 'normalized':
        D = sparse.lil_matrix(G.W.sum().diagonal() ** (-0.5))
        L = sparse.lil_matrix(np.matlib.identity(G.N)) - D * G.W * D
    if G.gtype == 'none':
        L = sparse.lil_matrix(0)
    else:
        raise AttributeError('Unknown laplacian type!')
    return L
