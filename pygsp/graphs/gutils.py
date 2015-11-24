# -*- coding: utf-8 -*-

from pygsp import utils

import numpy as np
import scipy as sp
from scipy import sparse
from math import isinf, isnan

logger = utils.build_logger(__name__)


def check_weights(W):
    r"""
    Check the characteristics of the weights matrix.

    Parameters
    ----------
    W : weights matrix
        The weights matrix to check

    Returns
    -------
    A dict of bools containing informations about the matrix

    has_inf_val : bool
        True if the matrix has infinite values else false
    has_nan_value : bool
        True if the matrix has a not a number value else false
    is_not_square : bool
        True if the matrix is not square else false
    diag_is_not_zero : bool
        True if the matrix diagonal has not only zero value else false

    Examples
    --------
    >>> from scipy import sparse
    >>> from pygsp.graphs import gutils
    >>> np.random.seed(42)
    >>> W = sparse.rand(10,10,0.2)
    >>> weights_chara = gutils.check_weights(W)

    """

    has_inf_val = False
    diag_is_not_zero = False
    is_not_square = False
    has_nan_value = False

    if isinf(W.sum()):
        logger.warning("GSP_TEST_WEIGHTS: There is an infinite "
                       "value in the weight matrix")
        has_inf_val = True

    if abs(W.diagonal()).sum() != 0:
        logger.warning("GSP_TEST_WEIGHTS: The main diagonal of "
                       "the weight matrix is not 0!")
        diag_is_not_zero = True

    if W.get_shape()[0] != W.get_shape()[1]:
        logger.warning("GSP_TEST_WEIGHTS: The weight matrix is "
                       "not square!")
        is_not_square = True

    if isnan(W.sum()):
        logger.warning("GSP_TEST_WEIGHTS: There is an NaN "
                       "value in the weight matrix")
        has_nan_value = True

    return {'has_inf_val': has_inf_val,
            'has_nan_value': has_nan_value,
            'is_not_square': is_not_square,
            'diag_is_not_zero': diag_is_not_zero}


def symmetrize(W, symmetrize_type='average'):
    r"""
    Symetrize a matrix.

    Parameters
    ----------
    W : sparse matrix
        Weight matrix
    symmetrize_type : string
        type of symmetrization (default 'average')
        The availlable symmetrization types are:
        'average' : average of W and W^T (default)
        'full'    : copy the missing entries

    Returns
    -------
    W : sparse matrix
        symmetrized matrix

    Examples
    --------
    >>> from pygsp.graphs import gutils
    >>> import numpy as np
    >>> from scipy import sparse
    >>> x = sparse.coo_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]))
    >>> W2 = gutils.symmetrize(x)
    >>> W1 = gutils.symmetrize(x, symmetrize_type='full')

    """

    if symmetrize_type == 'average':
        W = (W + W.T) / 2.

    elif symmetrize_type == 'full':
        A = W > 0
        M = (A - (A.T * A))
        W = sparse.csr_matrix(W.T)
        W[M.T] = W.T[M.T]

    else:
        raise ValueError("Unknown symmetrize type")

    return W
