# -*- coding: utf-8 -*-

r"""
Module documentation.
"""

import numpy as np

class Graph(object):

    def __init__(self, W, A, N, d, Ne, gtype, directed, lap_type, L, k=12, **kwargs):
        self.M = N
        pass

    def copy_graph_attr(self, gtype, Gn):
        pass
    def separate_graph(self):
        pass
    def subgraph(self, c):
        pass


class Grid2d(Graph):

    def __init__(self, M):
        super(Grid2d, self).__init__(**kwargs)
        pass

class 

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
