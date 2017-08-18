# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.utils` module implements some utility functions used throughout
the package.
"""

from __future__ import division

import sys
import importlib
import logging
from functools import wraps
from pkgutil import get_data
from io import BytesIO

import numpy as np
from scipy import kron, ones
from scipy import sparse
from scipy.io import loadmat as scipy_loadmat


def build_logger(name, **kwargs):
    logger = logging.getLogger(name)

    logging_level = kwargs.pop('logging_level', logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s:[%(levelname)s](%(name)s.%(funcName)s): %(message)s")

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
    @wraps(func)

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
    >>> from pygsp.utils import loadmat
    >>> data = loadmat('pointclouds/bunny')
    >>> data['bunny'].shape
    (2503, 3)

    """
    data = get_data('pygsp', 'data/' + path + '.mat')
    data = BytesIO(data)
    return scipy_loadmat(data)


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
    >>> import numpy as np
    >>> from pygsp.utils import distanz
    >>> x = np.arange(3)
    >>> distanz(x, x)
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

    d = abs(kron(ones((cy, 1)), xx).T +
            kron(ones((cx, 1)), yy) - 2 * xy)

    return np.sqrt(d)


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


def symmetrize(W, symmetrize_type='average'):
    r"""
    Symmetrize a square matrix.

    Parameters
    ----------
    W : array_like
        Square matrix to be symmetrized
    symmetrize_type : string
        'average' : symmetrize by averaging with the transpose.
        'full' : symmetrize by filling in the holes in the transpose.

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp.utils import symmetrize
    >>> x = np.array([[1,0],[3,4.]])
    >>> x
    array([[ 1.,  0.],
           [ 3.,  4.]])
    >>> symmetrize(x)
    array([[ 1. ,  1.5],
           [ 1.5,  4. ]])
    >>> symmetrize(x, symmetrize_type='full')
    array([[ 1.,  3.],
           [ 3.,  4.]])

    """
    if W.shape[0] != W.shape[1]:
        raise ValueError("Matrix must be square")

    sparse_flag = True if sparse.issparse(W) else False

    if symmetrize_type == 'average':
        return (W + W.T) / 2.
    elif symmetrize_type == 'full':
        A = (W > 0)
        if sparse_flag:
            mask = ((A + A.T) - A).astype('float')
        else:
            # numpy boolean subtract is deprecated in python 3
            mask = np.logical_xor(np.logical_or(A, A.T), A).astype('float')
        W += mask.multiply(W.T) if sparse_flag else (mask * W.T)
        return (W + W.T) / 2.  # Resolve ambiguous entries
    else:
        raise ValueError("Unknown symmetrization type.")


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
    >>> import pygsp
    >>> x = np.array([[1, 6], [2, 5], [3, 4]])
    >>> pygsp.utils.rescale_center(x)
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
        Largest eigenvalue, i.e. :py:attr:`~pygsp.graphs.Graph.lmax`.
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
