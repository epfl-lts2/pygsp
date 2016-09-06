# -*- coding: utf-8 -*-
r"""This module implements some utilitary functions used throughout the PyGSP box."""

import numpy as np
import scipy as sp
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

    d = abs(sp.kron(sp.ones((cy, 1)), xx).T +
            sp.kron(sp.ones((cx, 1)), yy) - 2*xy)

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

    Reference
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
