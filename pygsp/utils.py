# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
from math import isinf, isnan

import pygsp


def graph_array_handler(func):

    def inner(G, *args, **kwargs):

        if isinstance(G, pygsp.graphs.Graph):
            return func(G, *args, **kwargs)

        elif type(G) is list:
            output = []
            for g in G:
                output.append(func(g, *args, **kwargs))

            return output

        else:
            raise TypeError("This function only accept Graphs or Graphs lists")

    return inner


def filterbank_handler(func):

    def inner(f, *args, **kwargs):
        if len(f.g) <= 1:
            return func(f, *args, **kwargs)
        elif len(f.g) > 1:
            output = []
            i = range(len(f.g))
            for ii in i:
                output.append(func(f, *args, i=ii, **kwargs))
            return output

        else:
            raise TypeError("This function only accepts Filters or\
                            Filters lists")
    return inner


def sparsifier(func):

    def inner(*args, **kwargs):
        return sparse.lil_matrix(func(*args, **kwargs))

    return inner


def is_directed(M):
    r"""
    Returns a bool:  True if the graph is directed and false if not

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
    >>> from pygsp import graphs, utils
    >>> W = sparse.rand(10,10,0.2)
    >>> G = graphs.Graph(W=W)
    >>> is_directed = utils.is_directed(G.W)

    """
    # To pass a graph or a weight matrix as an argument
    if isinstance(M, pygsp.graphs.Graph):
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

    >>> import pygsp
    >>> import numpy as np
    >>> W = np.arange(16).reshape(4, 4)
    >>> G = pygsp.graphs.Graph(W)
    >>> lmax = pygsp.utils.estimate_lmax(G)

    """
    try:
        lmax = sparse.linalg.eigs(G.L, k=1, tol=5e-3, ncv=10)[0]
        # MAT: lmax=eigs(G.L,1,'lm',opts)
        # On robustness purposes, increasing the error by 1 percent
        lmax *= 1.01
    except ValueError:
        if G.verbose:
            print('GSP_ESTIMATE_LMAX: Cannot use default method')
        lmax = np.max(G.d)
    G.lmax = np.real(lmax)
    return np.real(lmax)


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
        GSP_TEST_WEIGHTS: The main diagonal of the weight matrix is not 0!
    >>> weights_chara = utils.check_weights(W)
    GSP_TEST_WEIGHTS: The main diagonal of the weight matrix is not 0!

    """

    has_inf_val = False
    diag_is_not_zero = False
    is_not_square = False
    has_nan_value = False

    if isinf(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an infinite "
              "value in the weight matrix")
        has_inf_val = True

    if abs(W.diagonal()).sum() != 0:
        print("GSP_TEST_WEIGHTS: The main diagonal of "
              "the weight matrix is not 0!")
        diag_is_not_zero = True

    if W.get_shape()[0] != W.get_shape()[1]:
        print("GSP_TEST_WEIGHTS: The weight matrix is "
              "not square!")
        is_not_square = True

    if isnan(W.sum()):
        print("GSP_TEST_WEIGHTS: There is an infinite "
              "value in the weight matrix")
        has_nan_value = True

    return [has_inf_val, has_nan_value, is_not_square, diag_is_not_zero]


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
    is_connected : bool
        A bool value telling if the graph is connected

    """

    A = G.W
    if not hasattr(G, 'directed'):
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
        is_connected = True

    return is_connected, in_conn, out_conn


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
    xy = np.dot(np.transpose(x), y)

    d = abs(sp.kron(sp.ones((cy, 1)), xx).transpose() +
            sp.kron(sp.ones((cx, 1)), yy) - 2*xy)

    return np.sqrt(d)


def repmatline(A, ncol=1, nrow=1):
    r"""
    This function repeat the matrix A in a specific manner

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

    return np.repeat(np.repeat(x, ncol, axis=1), nrow, axis=0)


def resistance_distance(G):
    r"""
    Compute the resitance distances of a graph

    Parameters
    ----------
    G : Graph structure or Laplacian matrix (L)
    verbose (bool) : Display parameter - False no log - True display warnings
        Default is True

    Returns
    -------
    rd : distance matrix

    Examples
    --------
    >>>
    >>>
    >>>

    Reference
    ----------
    :cite:`klein1993resistance`


    """
    if isinstance(G, pygsp.graphs.Graph):
        if not G.lap_type == 'combinatorial':
            pygsp.operators.create_laplacian(G, lap_type='combinatorial', get_laplacian_only=False)
            if verbose:
                print('Compute the combinatorial laplacian for the resitance distance')
        L = G.L

    else:
        L = G

    pseudo = np.linalg.pinv(L.todense())
    N = np.shape(L)[0]

    d = np.diagonal(pseudo)
    rd = np.kron(np.ones((1, N)), d) + np.kron(np.ones((N, 1)), d) - pseudo - pseudo.transpose()

    return rd


def symetrize(W, symetrize_type='average'):
    r"""
    symetrize a matrix

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
        W : symetrized matrix


    Examples
    --------
    >>> from pygsp import utils
    >>> W = utils.symetrize(W)
    >>> W = utils.symetrize(W, symetrize_type='average')

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


def tree_depths(A, root):

    if check_connectivity(A) == 0:
        raise ValueError('Graph is not connected')

    N = np.shape(A)[0]
    assigned = root-1
    depths = np.zeros((N))
    parents = np.zeros((N))

    next_to_expand = np.array([root])
    current_depth = 1

    while len(assigned) < N:
        new_entries_whole_round = []
        for i in range(len(next_to_expand)):
            neighbors = np.where(A[next_to_expand[i]] > 1e-7)[0]
            new_entries = np.setdiff1d(neighbors, assigned)
            parents[new_entries] = next_to_expand[i]
            depths[new_entries] = current_depth
            assigned = np.concatenate((assigned, new_entries))
            new_entries_whole_round = np.concatenate((new_entries_whole_round,
                                                      new_entries))

        current_depth = current_depth+1
        next_to_expand = new_entries_whole_round

    return depths, parents


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
