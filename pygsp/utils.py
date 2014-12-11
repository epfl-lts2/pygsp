import numpy as np
import scipy as sp
from scipy import sparse
from math import isinf, isnan


def is_directed(G):
    r"""
    Determine wether the graph is directed or not

    Parameters
    ----------
    G : graph
        The graph to determine

    Returns
    -------
    is_dir : bool
        True if the graph is directed and false if not

    Comments
    --------
    The Weight matrix has to be sparse (For now)
    Can also be used to check if a matrix is symetrical

    Examples
    --------
    >>> from scipy import sparse
    >>> from pygsp import graphs, utils
    >>> W = sparse.rand(10,10,0.2)
    >>> G = graphs.Graph(W=W)
    >>> is_directed = utils.is_directed(G.W)
    """

    is_dir = (G.W - G.W.transpose()).sum() != 0
    return is_dir


def estimate_lmax(G):
    r"""
    Estimate lmax at first by trying to take the first eigenvalue, and 
    if it doesn't work by using the greatest value of the degree vector

    Parameters
    ----------
    G : graph
        The graph on to determine

    Returns
    -------
    lmax : float
        Returns the value of lmax

    Examples
    --------
    >>> from pygsp import graphs, utils
    # TODO end doc
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
    Check the charasteristics of the weights matrix

    Parameters
    ----------
    W : weights matrix
        The weights matrix to check

    Returns
    -------
    An array of bool containing informations about the matrix

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
    >>> from pygsp import graphs, utils
    >>> W = sparse.rand(10,10,0.2)
    >>> [has_inf_val, has_nan_value, is_not_square, diag_is_not_zero] = utils.check_weights(W)
    or
    >>> weights_chara = utils.check_weights(W)
    """

    has_inf_val = False
    diag_is_not_zero = False
    is_not_square = False
    has_nan_value = False
    if isinf(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an infinite \
              value in the weight matrix")
        has_inf_val = True
    if abs(W.diagonal()).sum() != 0:
        print("GSP_TEST_WEIGHTS: The main diagonal of \
              the weight matrix is not 0!")
        diag_is_not_zero = True
    if W.get_shape()[0] != W.get_shape()[1]:
        print("GSP_TEST_WEIGHTS: The weight matrix is \
              not square!")
        is_not_square = True
    if isnan(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an infinite \
              value in the weight matrix")
        has_nan_value = True

    return [has_inf_val, has_nan_value, is_not_square, diag_is_not_zero]


def create_laplacian(G):
    r"""
    Create the laplacian of a graph from it's weights matrix and the laplacian's type

    Parameters
    ----------
    G : graph
        The graph wich will be used to create the laplacian

    Returns
    -------
    L : sparse.lil_matrix
        The laplacian under the form of a sparse matrix 

    Examples
    --------
    >>> from pygsp import graphs, utils
    >>> G = graphs.Graph()
    >>> L = utils.create_laplacian(G)
    """
    if sp.shape(G.W) == (1,1):
        return sparse.lil_matrix(0)
    else:
        if G.lap_type == 'combinatorial':
            L = sparse.lil_matrix(G.W.sum().diagonal() - G.W)
        if G.lap_type == 'normalized':
            D = sparse.lil_matrix(G.W.sum().diagonal() ** (-0.5))
            L = sparse.lil_matrix(np.matlib.identity(G.N)) - D * G.W * D
        if G.lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')
        return L


def check_connectivity(G, **kwargs):
    r"""
    Function to check the connectivity of the input graph
    It will call _check_connectivity_directed or _check_connectivity_undirected 
    wether the graph is directed or not

    Parameters
    ----------
    G : graph
        Graph to check
    **kwargs : keyowords arguments
        (not implmented yet)

    Returns
    -------
    is_connected = bool
        A bool value telling if the graph is connected


    """

    A = G.W
    # Removing the diagonal
    A -= A.diagonal()
    if G.directed:
        return _check_connectivity_directed(A, kwargs)
    else:
        return _check_connectivity_undirected(A, kwargs)
    pass


def _check_connectivity_directed(A, **kwargs):
    is_connected = (A.W <= 0).all()
    for c in sp.shape(A.W)[0]:
        c_is_connected = (c == 0).all()
        if c_is_connected:
            break
    for r in sp.shape(A.W)[1]:
        r_is_connected = (c == 0).all()
        if r_is_connected:
            break
    # TODO check axises
    in_conn = (A.sum(axis=1)>0).nonzeros()
    out_conn = (A.sum(axis=2)>0).nonzeros()

    if c_is_connected and r_is_connected:
        is_connected = True
 
    return is_connected, in_conn, out_conn


def _check_connectivity_undirected(A, **kwargs):
    is_connected = (A.W <= 0).all()
    for c in sp.shape(A.W)[0]:
        c_is_connected = (c == 0).all()
        if c_is_connected:
            break
    # TODO check axises
    in_conn = (A.sum(axis=1)>0).nonzeros()
    out_conn = in_conn
    if c_is_connected and r_is_connected:
        is_connected = True
 
    return is_connected, in_conn, out_conn
        

def distanz(x, y=None):
    r"""
    Calculate the distanz between two colon vectors

    Parameters
    ----------
    x = ndarray
        First colon vector
    y = ndarray
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
    if y is None:
        y = x

    rx, cx = x.shape()
    ry, cy = y.shape()

    # Size verification
    if rx != ry:
        raise("The sizes of x and y do not fit")
    xx = (x**x).sum()
    yy = (y**y).sum()
    xy = np.transpose(x)*y
    d = abs(sp.kron(sp.ones((1, cy)), xx) + sp.kron(sp.ones((cx, 1)), yy) - 2*xy)
    return d


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
