# -*- coding: utf-8 -*-
r"""This module implements some utilitary functions used throughout the PyGSP box."""

import numpy as np
from scipy import kron, ones
from scipy import sparse
import logging


def build_logger(name, **kwargs):
    logger = logging.getLogger(name)

    logging_level = kwargs.pop('logging_level', logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s:[%(levelname)s](%(name)s.%(funcName)s): %(message)s")

        steam_handler = logging.StreamHandler()
        steam_handler.setLevel(logging_level)
        steam_handler.setFormatter(formatter)

        logger.setLevel(logging_level)
        logger.addHandler(steam_handler)

    return logger


logger = build_logger(__name__)


def graph_array_handler(func):

    def inner(G, *args, **kwargs):

        if type(G) is list:
            output = []
            for g in G:
                output.append(func(g, *args, **kwargs))

            return output

        else:
            return func(G, *args, **kwargs)

    return inner


def filterbank_handler(func):

    def inner(f, *args, **kwargs):
        if 'i' in kwargs:
            return func(f, *args, **kwargs)

        if len(f.g) <= 1:
            return func(f, *args, **kwargs)
        elif len(f.g) > 1:
            output = []
            for i in range(len(f.g)):
                output.append(func(f, *args, i=i, **kwargs))
            return output
    return inner


def sparsifier(func):

    def inner(*args, **kwargs):
        return sparse.lil_matrix(func(*args, **kwargs))

    return inner


def pyunlocbox_required(func):

    def inner(*args, **kwargs):
        try:
            import pyunlocbox
        except ImportError:
            logger.error('Cannot import pyunlocbox')
        return func(*args, **kwargs)


def distanz(x, y=None):
    r"""
    Calculate the distanz between two colon vectors

    Parameters
    ----------
    x : ndarray
        First colon vector
    y : ndarray
        Second colon vector

    Returns
    -------
    d : ndarray
        Distance between x and y

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import utils
    >>> x = np.random.rand(16)
    >>> y = np.random.rand(16)
    >>> distanz = utils.distanz(x, y)

    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(1, x.shape[0])

    if y is None:
        y = x

    else:
        try:
            y.shape[1]
        except IndexError:
            y = y.reshape(1, y.shape[0])

    rx, cx = x.shape
    ry, cy = y.shape

    # Size verification
    if rx != ry:
        raise("The sizes of x and y do not fit")

    xx = (x*x).sum(axis=0)
    yy = (y*y).sum(axis=0)
    xy = np.dot(x.T, y)

    d = abs(kron(ones((cy, 1)), xx).T +
            kron(ones((cx, 1)), yy) - 2*xy)

    return np.sqrt(d)


def resistance_distance(M):  # 1 call dans operators.reduction
    r"""
    Compute the resistance distances of a graph.

    Parameters
    ----------
    M : Graph or sparse matrix
        Graph structure or Laplacian matrix (L)

    Returns
    -------
    rd : sparse matrix
        distance matrix

    Examples
    --------
    >>>
    >>>
    >>>

    References
    ----------
    :cite:`klein1993resistance`

    """
    if sparse.issparse(M):
        L = M.tocsc()

    else:
        if not M.lap_type == 'combinatorial':
            logger.info('Compute the combinatorial laplacian for the resitance'
                        ' distance')
            M.create_laplacian(lap_type='combinatorial')
        L = M.L.tocsc()

    try:
        pseudo = sparse.linalg.inv(L)
    except RuntimeError:
        pseudo = sparse.lil_matrix(np.linalg.pinv(L.toarray()))

    N = np.shape(L)[0]
    d = sparse.csc_matrix(pseudo.diagonal())
    rd = sparse.kron(d, sparse.csc_matrix(np.ones((N, 1)))).T \
        + sparse.kron(d, sparse.csc_matrix(np.ones((N, 1)))) \
        - pseudo - pseudo.T

    return rd

def approx_resistance_distance(g, epsilon):
    r"""
    Compute the resistance distances of each edge of a graph using the
    Spielman-Srivastava algorithm.

    Parameters
    ----------
    g : Graph
        Graph structure

    epsilon: float
        Sparsification parameter

    Returns
    -------
    rd : ndarray
        distance for every edge in the graph

    Examples
    --------
    >>>
    >>>
    >>>

    Notes
    -----
    This implementation avoids the blunt matrix inversion of the exact distance
    distance and can scale to very large graphs. The approximation error is
    included in the budget of Spielman-Srivastava sparsification algorithm.

    References
    ----------
    :cite:`klein1993resistance` :cite:`spielman2011graph`

    """
    g.create_incidence_matrix()
    n = g.N
    k = 24 * np.log( n / epsilon)
    Q = ((np.random.rand(int(k),g.Wb.shape[0]) > 0.5)*2. -1)/np.sqrt(k)
    Y = sparse.csc_matrix(Q).dot(np.sqrt(g.Wb).dot(g.B))

    r = splu_inv_dot(g.L, Y.T)

    return ((r[g.start_nodes] - r[g.end_nodes]).toarray()**2).sum(axis=1)

def extract_submatrix(M, ind_rows, ind_cols):
    r"""
    Extract a bloc of specific rows and columns from a sparse matrix.

    Parameters
    ----------

    M : sparse matrix
        Input matrix

    ind_rows: ndarray
        Indices of rows to extract

    ind_cols: ndarray
        Indices of columns to extract

    Returns
    -------

    sub_M: sparse matrix
        Submatrix obtained from M keeping only the requested rows and columns

    Examples
    --------
    >>> # Extracting first diagonal block from a sparse matrix
    >>> M = sparse.csc_matrix((16, 16))
    >>> ind_row = range(8); ind_col = range(8)
    >>> block = extract_submatrix(M, ind_row, ind_col)
    >>> block.shape
    (8, 8)
    """
    M = M.tocoo()

    # Finding elements of the sub-matrix
    m = np.in1d(M.row, ind_rows) & np.in1d(M.col, ind_cols)
    n_elem = m.sum()

    # Finding new rows and column indices
    # The concatenation with ind and ind_comp is there to account for the fact that some rows
    # or columns may not have elements in them, which foils this np.unique trick
    _, row = np.unique(np.concatenate([M.row[m], ind_rows]), return_inverse=True)
    _, col = np.unique(np.concatenate([M.col[m], ind_cols]), return_inverse=True)

    return sparse.coo_matrix((M.data[m], (row[:n_elem],col[:n_elem])),
                         shape=(len(ind_rows),len(ind_cols)),copy=True)


def splu_inv_dot(A, B, threshold=np.spacing(1)):
    """
    Compute A^{-1}B for sparse matrix A assuming A is Symmetric Diagonally
    Dominant (SDD).

    Parameters
    ----------
    A : sparse matrix
        Input SDD matrix to invert, in CSC or CSR form.

    B : sparse matrix
        Matrix or vector of the right hand side

    threshold: float, optional
        Threshold to apply to result as to remove numerical noise before
        conversion to sparse format. (default: machine precision)

    Returns
    -------
    res: sparse matrix
        Result of A^{-1}B

    Notes
    -----
    This inversion by sparse linear system solving is optimized for SDD matrices
    such as Graph Laplacians. Note that B is converted to a dense matrix before
    being sent to splu, which is more computationally efficient but can lead to
    very large memory usage if B is large.
    """
    # Compute the LU decomposition of A
    lu = sparse.linalg.splu(A,
                        diag_pivot_thresh=A.diagonal().min()*0.5,
                        permc_spec='MMD_AT_PLUS_A',
                        options={'SymmetricMode':True})

    res = lu.solve(B.toarray())

    # Threshold the result to remove numerical noise
    res[abs(res) < threshold] = 0

    # Convert to sparse matrix
    res = sparse.csc_matrix(res)

    return res
