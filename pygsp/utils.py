# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.utils` module implements some utility functions used throughout
the package.
"""

from __future__ import division

import sys
import importlib
import logging
import functools
import pkgutil
import io

import numpy as np
from scipy import sparse
import scipy.io


def build_logger(name):
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s:[%(levelname)s](%(name)s.%(funcName)s): %(message)s")

        steam_handler = logging.StreamHandler()
        steam_handler.setLevel(logging.DEBUG)
        steam_handler.setFormatter(formatter)

        logger.setLevel(logging.DEBUG)
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

    # Preserve documentation of func.
    @functools.wraps(func)

    def inner(f, *args, **kwargs):

        if 'i' in kwargs:
            return func(f, *args, **kwargs)

        elif f.Nf <= 1:
            return func(f, *args, **kwargs)

        else:
            output = []
            for i in range(f.Nf):
                output.append(func(f, *args, i=i, **kwargs))
            return output

    return inner


def sparsifier(func):

    def inner(*args, **kwargs):
        return sparse.lil_matrix(func(*args, **kwargs))

    return inner


def loadmat(path):
    r"""
    Load a matlab data file.

    Parameters
    ----------
    path : string
        Path to the mat file from the data folder, without the .mat extension.

    Returns
    -------
    data : dict
        dictionary with variable names as keys, and loaded matrices as
        values.

    Examples
    --------
    >>> from pygsp import utils
    >>> data = utils.loadmat('pointclouds/bunny')
    >>> data['bunny'].shape
    (2503, 3)

    """
    data = pkgutil.get_data('pygsp', 'data/' + path + '.mat')
    data = io.BytesIO(data)
    return scipy.io.loadmat(data)


def distanz(x, y=None):
    r"""
    Calculate the distance between two colon vectors.

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
    >>> from pygsp import utils
    >>> x = np.arange(3)
    >>> utils.distanz(x, x)
    array([[ 0.,  1.,  2.],
           [ 1.,  0.,  1.],
           [ 2.,  1.,  0.]])

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
        raise ValueError("The sizes of x and y do not fit")

    xx = (x * x).sum(axis=0)
    yy = (y * y).sum(axis=0)
    xy = np.dot(x.T, y)

    d = abs(np.kron(np.ones((cy, 1)), xx).T +
            np.kron(np.ones((cx, 1)), yy) - 2 * xy)

    return np.sqrt(d)


def resistance_distance(G):
    r"""
    Compute the resistance distances of a graph.

    Parameters
    ----------
    G : Graph or sparse matrix
        Graph structure or Laplacian matrix (L)

    Returns
    -------
    rd : sparse matrix
        distance matrix

    References
    ----------
    :cite:`klein1993resistance`
    """

    if sparse.issparse(G):
        L = G.tocsc()

    else:
        if G.lap_type != 'combinatorial':
            raise ValueError('Need a combinatorial Laplacian.')
        L = G.L.tocsc()

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


def symmetrize(W, method='average'):
    r"""
    Symmetrize a square matrix.

    Parameters
    ----------
    W : array_like
        Square matrix to be symmetrized
    method : string
        * 'average' : symmetrize by averaging with the transpose. Most useful
          when transforming a directed graph to an undirected one.
        * 'maximum' : symmetrize by taking the maximum with the transpose.
          Similar to 'fill' except that ambiguous entries are resolved by
          taking the largest value.
        * 'fill' : symmetrize by filling in the zeros in both the upper and
          lower triangular parts. Ambiguous entries are resolved by averaging
          the values.
        * 'tril' : symmetrize by considering the lower triangular part only.
        * 'triu' : symmetrize by considering the upper triangular part only.

    Examples
    --------
    >>> from pygsp import utils
    >>> W = np.array([[0, 3, 0], [3, 1, 6], [4, 2, 3]], dtype=float)
    >>> W
    array([[ 0.,  3.,  0.],
           [ 3.,  1.,  6.],
           [ 4.,  2.,  3.]])
    >>> utils.symmetrize(W, method='average')
    array([[ 0.,  3.,  2.],
           [ 3.,  1.,  4.],
           [ 2.,  4.,  3.]])
    >>> utils.symmetrize(W, method='maximum')
    array([[ 0.,  3.,  4.],
           [ 3.,  1.,  6.],
           [ 4.,  6.,  3.]])
    >>> utils.symmetrize(W, method='fill')
    array([[ 0.,  3.,  4.],
           [ 3.,  1.,  4.],
           [ 4.,  4.,  3.]])
    >>> utils.symmetrize(W, method='tril')
    array([[ 0.,  3.,  4.],
           [ 3.,  1.,  2.],
           [ 4.,  2.,  3.]])
    >>> utils.symmetrize(W, method='triu')
    array([[ 0.,  3.,  0.],
           [ 3.,  1.,  6.],
           [ 0.,  6.,  3.]])

    """
    if W.shape[0] != W.shape[1]:
        raise ValueError('Matrix must be square.')

    if method == 'average':
        return (W + W.T) / 2

    # Sum is 2x average. It is not a good candidate as it modifies an already
    # symmetric matrix.

    elif method == 'maximum':
        if sparse.issparse(W):
            bigger = (W.T > W)
            return W - W.multiply(bigger) + W.T.multiply(bigger)
        else:
            return np.maximum(W, W.T)

    elif method == 'fill':
        A = (W > 0)  # Boolean type.
        if sparse.issparse(W):
            mask = (A + A.T) - A
            W = W + mask.multiply(W.T)
        else:
            # Numpy boolean subtract is deprecated.
            mask = np.logical_xor(np.logical_or(A, A.T), A)
            W = W + mask * W.T
        return symmetrize(W, method='average')  # Resolve ambiguous entries.

    elif method in ['tril', 'triu']:
        if sparse.issparse(W):
            tri = getattr(sparse, method)
        else:
            tri = getattr(np, method)
        W = tri(W)
        return symmetrize(W, method='maximum')

    else:
        raise ValueError('Unknown symmetrization method {}.'.format(method))


def rescale_center(x):
    r"""
    Rescale and center data, e.g. embedding coordinates.

    Parameters
    ----------
    x : ndarray
        Data to be rescaled.

    Returns
    -------
    r : ndarray
        Rescaled data.

    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.array([[1, 6], [2, 5], [3, 4]])
    >>> utils.rescale_center(x)
    array([[-1. ,  1. ],
           [-0.6,  0.6],
           [-0.2,  0.2]])

    """
    N = x.shape[1]
    y = x - np.kron(np.ones((1, N)), np.mean(x, axis=1)[:, np.newaxis])
    c = np.amax(y)
    r = y / c

    return r


def compute_log_scales(lmin, lmax, Nscales, t1=1, t2=2):
    r"""
    Compute logarithm scales for wavelets.

    Parameters
    ----------
    lmin : float
        Smallest non-zero eigenvalue.
    lmax : float
        Largest eigenvalue, i.e. :py:attr:`pygsp.graphs.Graph.lmax`.
    Nscales : int
        Number of scales.

    Returns
    -------
    scales : ndarray
        List of scales of length Nscales.

    Examples
    --------
    >>> from pygsp import utils
    >>> utils.compute_log_scales(1, 10, 3)
    array([ 2.       ,  0.4472136,  0.1      ])

    """
    scale_min = t1 / lmax
    scale_max = t2 / lmin
    return np.exp(np.linspace(np.log(scale_max), np.log(scale_min), Nscales))


def repmatline(A, ncol=1, nrow=1):
    r"""
    Repeat the matrix A in a specific manner.

    Parameters
    ----------
    A : ndarray
    ncol : int
        default is 1
    nrow : int
        default is 1

    Returns
    -------
    Ar : ndarray

    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.array([[1, 2], [3, 4]])
    >>> x
    array([[1, 2],
           [3, 4]])
    >>> utils.repmatline(x, nrow=2, ncol=3)
    array([[1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4],
           [3, 3, 3, 4, 4, 4]])

    """

    if ncol < 1 or nrow < 1:
        raise ValueError('The number of lines and rows must be greater or '
                         'equal to one, or you will get an empty array.')

    return np.repeat(np.repeat(A, ncol, axis=1), nrow, axis=0)


def import_modules(names, src, dst):
    """Import modules in package."""
    for name in names:
        module = importlib.import_module(src + '.' + name)
        setattr(sys.modules[dst], name, module)


def import_classes(names, src, dst):
    """Import classes in package from their implementation modules."""
    for name in names:
        module = importlib.import_module('pygsp.' + src + '.' + name.lower())
        setattr(sys.modules['pygsp.' + dst], name, getattr(module, name))


def import_functions(names, src, dst):
    """Import functions in package from their implementation modules."""
    for name in names:
        module = importlib.import_module('pygsp.' + src)
        setattr(sys.modules['pygsp.' + dst], name, getattr(module, name))
