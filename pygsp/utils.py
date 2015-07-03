# -*- coding: utf-8 -*-
r"""
This module implements some utilitary functions used throughout the PyGSP box.
"""

import numpy as np
import scipy as sp
from scipy import sparse
import logging


def build_logger(name):
    logger = logging.getLogger(name)

    formatter = logging.Formatter("%(asctime)s:[%(levelname)s](%(name)s.%(funcName)s): %(message)s")

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    steam_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger


logger = build_logger(__name__)


def graph_array_handler(func):

    def inner(G, *args, **kwargs):

        from pygsp.graphs import Graph

        if isinstance(G, Graph):
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
        if hasattr(f.g, '__call__'):
            return func([f], *args, **kwargs)
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

    d = abs(sp.kron(sp.ones((cy, 1)), xx).T +
            sp.kron(sp.ones((cx, 1)), yy) - 2*xy)

    return np.sqrt(d)


def full_eigen(L):
    r"""
    Computes full eigen decomposition on a matrix

    Parameters
    ----------
    L : ndarray
        Matrix to decompose

    Returns
    -------
    EVa : ndarray
        Eigenvalues
    EVe : ndarray
        Eigenvectors

    """

    eigenvectors, eigenvalues, _ = np.linalg.svd(L.todense())

    # Sort everything

    inds = np.argsort(eigenvalues)
    EVa = np.sort(eigenvalues)

    # TODO check if axis are good
    EVe = eigenvectors[:, inds]

    for val in EVe[0, :].reshape(EVe.shape[0], 1):
        if val < 0:
            val = -val

    return EVa, EVe


def resistance_distance(M):
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

    Reference
    ----------
    :cite:`klein1993resistance`


    """

    from pygsp.graphs import Graph
    from pygsp.operators import create_laplacian

    if isinstance(M, Graph):
        if not M.lap_type == 'combinatorial':
            logger.info('Compute the combinatorial laplacian for the resitance'
                        ' distance')
            create_laplacian(M, lap_type='combinatorial',
                             get_laplacian_only=False)
        L = M.L.tocsc()

    else:
        L = M.tocsc()

    try:
        pseudo = sparse.linalg.inv(L)
    except RuntimeError:
        pseudo = sparse.lil_matrix(np.linalg.pinv(L.toarray()))

    N = np.shape(L)[0]
    d = sparse.csc_matrix(pseudo.diagonal())
    rd = sparse.kron(d, sparse.csc_matrix(np.ones((N, 1)))).T + sparse.kron(d, sparse.csc_matrix(np.ones((N, 1)))) - pseudo - pseudo.T

    return rd


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
