# -*- coding: utf-8 -*-

from pygsp import utils

import numpy as np
import scipy as sp
from scipy import sparse
from math import isinf, isnan

logger = utils.build_logger(__name__)


def check_connectivity(G, **kwargs):
    r"""
    Function to check the connectivity of the input graph.
    It will call _check_connectivity_directed or _check_connectivity_undirected
    wether the graph is directed or not.

    Parameters
    ----------
    G : graph
        Graph to check
    **kwargs : keyowords arguments
        (not implmented yet)

    Returns
    -------
    is_connected : bool
        A bool value telling if the graph is connected
    in_conn : int
        Number of in connections
    out_conn : int
        Number of out connections

    """

    if not hasattr(G, 'directed'):
        G.directed = is_directed(G)
    # Removing the diagonal
    A = G.W - np.diag(G.W.diagonal())

    if G.directed:
        return _check_connectivity_directed(A, **kwargs)

    else:
        return _check_connectivity_undirected(A, **kwargs)


def _check_connectivity_directed(A, **kwargs):
    r"""
    Subfunc to check connec in the directed case.
    """
    is_connected = (A < 0).any()
    hard_check = (1 - (A.sum(axis=0) > 0)) +\
        (1 - (A.sum(axis=1) > 0)).reshape(1, A.shape[0])

    c = 0
    while c <= sp.shape(A)[0]:
        c_is_connected = (c == 0)
        c += 1
        if c_is_connected:
            break

    r = 0
    while r <= sp.shape(A)[1]:
        r_is_connected = (c == 0)
        r += 1
        if r_is_connected:
            break

    # TODO check axis
    in_conn = (1 - (A.sum(axis=0) > 0))
    out_conn = (1 - (A.sum(axis=1) > 0))

    return is_connected, in_conn, out_conn


def _check_connectivity_undirected(A, **kwargs):
    r"""
    Subfunc to check connec in the undirected case.
    """

    is_connected = (A < 0).any()
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
        return is_connected, in_conn, out_conn


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


@utils.graph_array_handler
def compute_fourier_basis(G, exact=None, cheb_order=30, **kwargs):
    r"""
    Compute the fourier basis of the graph G

    Parameters
    ----------
    G : Graph
        Graph structure

    Return
    ------
    G : Graph
        Graph structure modify

    Note
    ----
    'compute_fourier_basis(G)' computes a full eigendecomposition of the graph
    Laplacian G.L:

    .. L = U Lambda U*

    .. math:: {\cal L} = U \Lambda U^*

    where $\Lambda$ is a diagonal matrix of the Laplacian eigenvalues.
    *G.e* is a column vector of length *G.N* containing the Laplacian
    eigenvalues. The function will store the basis *U*, the eigenvalues
    *e*, the maximum eigenvalue *lmax* and *G.mu* the coherence of the
    Fourier basis into the structure *G*.

    Example
    -------
    >>> from pygsp import graphs
    >>> N = 50;
    >>> G = graphs.Sensor(N);
    >>> grahs.gutils.compute_fourier_basis(G);

    References
    ----------
    cite ´chung1997spectral´


    Author : David I Shuman, Nathanael Perraudin
    """

    if hasattr(G, 'e') or hasattr(G, 'U'):
        logger.error("This graph already has Laplacian eigenvectors or eigenvalues")
        return

    if G.N > 3000:
        logger.warning("Performing full eigendecomposition of a large matrix\
              may take some time.")

    if not hasattr(G, 'L'):
        raise AttributeError("Graph Laplacian is missing")
    G.e, G.U = utils.full_eigen(G.L)
    G.e = np.array(G.e)
    G.U = np.array(G.U)

    G.lmax = np.max(G.e)

    G.mu = np.max(np.abs(G.U))


@utils.graph_array_handler
def create_laplacian(G, lap_type=None, get_laplacian_only=True):
    r"""
    Create the graph laplacian of graph G

    Parameters
    ----------
    G : Graph
    lap_type : string :
        the laplacian type to use.
        Default is the lap_type attribute of G, otherwise it is "combinatorial".
    get_laplacian_only : bool
        True return each Laplacian in an array
        False set each Laplacian in each graphs.
        (default = True)

    Returns
    -------
    L : ndarray
        Laplacian matrix

    """
    if sp.shape(G.W) == (1, 1):
        return sparse.lil_matrix(0)

    if not lap_type:
        if not hasattr(G, 'lap_type'):
            lap_type = 'combinatorial'
            G.lap_type = lap_type
        else:
            lap_type = G.lap_type

    G.lap_type = lap_type

    if G.directed:
        if lap_type == 'combinatorial':
            L = 0.5*(sparse.diags(np.ravel(G.W.sum(0)), 0) + sparse.diags(np.ravel(G.W.sum(1)), 0) - G.W - G.W.getH()).tocsc()
        elif lap_type == 'normalized':
            raise NotImplementedError('Yet. Ask Nathanael.')
        elif lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')

    else:
        if lap_type == 'combinatorial':
            L = (sparse.diags(np.ravel(G.W.sum(1)), 0) - G.W).tocsc()
        elif lap_type == 'normalized':
            D = sparse.diags(np.ravel(np.power(G.W.sum(1), -0.5)), 0).tocsc()
            L = sparse.identity(G.N) - D * G.W * D
        elif lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')

    if get_laplacian_only:
        return L
    else:
        G.L = L


@utils.graph_array_handler
def estimate_lmax(G):
    r"""
    This function estimates lmax from a Graph object and stores it into the
    graph.

    Parameters
    ----------
    G : Graph object

    Returns
    -------
    lmax : float
        Returns the value of lmax

    Examples
    --------
    Just define a graph an apply the estimation on it

    >>> from pygsp import graphs
    >>> import numpy as np
    >>> W = np.arange(16).reshape(4, 4)
    >>> G = graphs.Graph(W)
    >>> lmax = graphs.gutils.estimate_lmax(G)
    >>> # or
    >>> graphs.gutils.estimate_lmax(G)
    """
    try:
        lmax = sparse.linalg.eigs(G.L, k=1, tol=5e-3, ncv=10)[0]
        # MAT: lmax=eigs(G.L,1,'lm',opts)
        # On robustness purposes, increasing the error by 1 percent
        lmax *= 1.01
    except ValueError:
        logger.warning('GSP_ESTIMATE_LMAX: Cannot use default method')
        lmax = np.max(G.d)
    G.lmax = np.real(lmax)
    return np.real(lmax)


def is_directed(M):
    r"""
    Returns a bool:  True if the graph is directed and false if not.

    Parameters
    ----------
    M : sparse matrix or Graph

    Returns
    -------
    is_dir : bool

    Notes
    -----
    The Weight matrix has to be sparse (For now)
    Can also be used to check if a matrix is symetrical

    Examples
    --------
    >>> from scipy import sparse
    >>> from pygsp import graphs
    >>> W = sparse.rand(10,10,0.2)
    >>> G = graphs.Graph(W=W)
    >>> is_directed = graphs.gutils.is_directed(G.W)
    """

    from pygsp.graphs import Graph

    # To pass a graph or a weight matrix as an argument
    if isinstance(M, Graph):
        W = M.W
    else:
        W = M

    # Python Bug Can't use this in tests
    if np.shape(W) != (1, 1):
        is_dir = np.abs((W - W.T)).sum() != 0
    else:
        is_dir = False

    return is_dir


def symetrize(W, symetrize_type='average'):
    r"""
    Symetrize a matrix.

    Parameters
    ----------
    W : sparse matrix
        Weight matrix
    symetrize_type : string
        type of symetrization (default 'average')
        The availlable symetrization_types are:
        'average' : average of W and W^T (default)
        'full'    : copy the missing entries
        'none'    : nothing is done (the matrix might stay unsymetric!)

    Returns
    -------
    W : sparse matrix
        symetrized matrix


    Examples
    --------
    >>> from pygsp.graphs import gutils
    >>> import numpy as np
    >>> from scipy import sparse
    >>> x = sparse.coo_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]))
    >>> W2 = gutils.symetrize(x)
    >>> W1 = gutils.symetrize(x, symetrize_type='average')

    """

    if symetrize_type == 'average':
        W = (W + W.getH())/2.
        return W

    elif symetrize_type == 'full':
        A = W > 0
        M = (A - (A.T * A))
        W = sparse.csr_matrix(W.T)
        W[M.T] = W.T[M.T]

        return W

    elif symetrize_type == 'none':
        return W

    else:
        raise ValueError("Unknown symetrize type")
