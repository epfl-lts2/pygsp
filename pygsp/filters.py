# -*- coding: utf-8 -*-

r"""
Flters Doc
"""

import numpy as np


class Filter(object):
    r"""
    TODO doc
    """

    def __init__(self):
        pass


class FilterBank(object):
    r"""
    A filterbank should just be a list of filter to apply
    """

    def __init__(self, F):
        self.F = F


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
    >>> pygsp.module1.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
