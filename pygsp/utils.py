import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
from math import isinf, isnan

import pygsp


def is_directed(M):
    r"""
    Returns a bool:  True if the graph is directed and false if not

    Parameters
    ----------
    M : sparse matrix or Graph

    Returns
    -------
    is_dir : bool

    Examples
    --------
    Just define a Graph and look if it is directed

    >>> import pygsp
    >>> G = pygsp.graph.Bunny()
    >>> pygsp.utils.is_directed(W)

    Notes
    -----
    The Weight matrix has to be sparse (For now)
    Can also be used to check if a matrix is symetrical
    """
    # To pass a graph or a weight matrix as an argument
    if issubclass(type(M), pygsp.graphs.Graph):
        W = M.W
    else:
        W = M

    # Python Bug Can't use this in tests
    if np.shape(W) != (1, 1):
        is_dir = (W - W.transpose()).sum() != 0
    else:
        is_dir = False
    return is_dir


def estimate_lmax(G):
    r"""
    This function estimates lmax from a Graph object

    Parameters
    ----------
    G : Graph object

    Returns
    -------
    lmax : float

    Examples
    --------
    Just define a graph an apply the estimation on it

    >>> import pygsp
    >>> G = pygsp.graph.Graph()
    >>> lmax = pygsp.utils.estimate_lmax(G)
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
    Check the values of a weight matrix

    Parameters
    ----------
    W : sparse matrix
        Weight matrix

    Returns
    -------
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
    if sp.shape(G.W) == (1, 1):
        return sparse.lil_matrix(0)
    else:
        if G.lap_type == 'combinatorial':
            L = sparse.lil_matrix(G.W.sum(1).diagonal() - G.W)
        elif G.lap_type == 'normalized':
            D = sparse.lil_matrix(G.W.sum(1).diagonal() ** (-0.5))
            L = sparse.lil_matrix(np.matlib.identity(G.N)) - D * G.W * D
        elif G.lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')
        return L


def check_connectivity(G, **kwargs):
    A = G.W
    try:
        G.directed
    except AttributeError:
        G.directed = is_directed(G)
    # Removing the diagonal
    A -= A.diagonal()
    if G.directed:
        return _check_connectivity_directed(A, **kwargs)
    else:
        return _check_connectivity_undirected(A, **kwargs)
    pass


def _check_connectivity_directed(A, **kwargs):
    is_connected = (A <= 0).all()
    c = 0
    while c <= sp.shape(A)[0]:
        c_is_connected = (c == 0).all()
        c += 1
        if c_is_connected:
            break
    r = 0
    while r <= sp.shape(A)[1]:
        r_is_connected = (c == 0).all()
        r += 1
        if r_is_connected:
            break
    # TODO check axises
    in_conn = (A.sum(axis=1) > 0).nonzero()
    out_conn = (A.sum(axis=2) > 0).nonzero()

    if c_is_connected and r_is_connected:
        return True, in_conn, out_conn


def _check_connectivity_undirected(A, **kwargs):
    is_connected = (A <= 0).all()
    c = 0
    while c <= sp.shape(A)[0]:
        c_is_connected = (c == 0)
        c += 1
        if c_is_connected:
            break
    # TODO check axises
    in_conn = (A.sum(axis=1) > 0).nonzero()
    out_conn = in_conn
    if c_is_connected:
        return True, in_conn, out_conn


def distanz(x, y=None):
    r"""
    paramters:
        - x: matrix with col vectors
        - y: matrix with col vectors
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
    xy = np.dot(np.transpose(x), y)
    d = abs(sp.kron(sp.ones((cy, 1)), xx).transpose() +
            sp.kron(sp.ones((cx, 1)), yy) - 2*xy)

    return np.sqrt(d)


def symetrize(W, symetrize_type='average'):
    r"""
    symetrize a matrix
    Usage:  W = gsp_symetrize(W)
            W = gsp_symetrize(W, symetrize_type='average')

    Input parameters:
        W: square matrix
        symetrize_type: type of symetrization (default 'average')

    Output parameters:
        W: symetrized matrix

    The availlable symetrization_types are:
        'average' : average of W and W^T (default)
        'full'    : copy the missing entries
        'none'    : nothing is done (the matrix might stay unsymetric!)
    """

    if symetrize_type == 'average':
        W = (W + W.getH())/2.
        return W

    elif symetrize_type == 'full':
        A = W > 0
        M = (A - (A.T.multiply(A)))
        W = sparse.csr_matrix(W)
        W[M.T] = W.T[M.T]
        return W

    elif symetrize_type == 'none':
        return W

    else:
        raise ValueError("Unknown symetrize type")


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a : int
        Description.
    b : array_like
        Description.
    c : bool
        Description.

    Returns
    -------
    d : ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.utils.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
