# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Grid2d(Graph):
    r"""
    Create a 2-dimensional grid graph.

    Parameters
    ----------
    shape : int or tuple, optional
        Dimensions of the 2-dimensional grid. Syntax: (height, width),
        (height,), or height, where the last two options imply width = height.
        Default is shape = (3,).

    Notes
    -----
    The total number of nodes on the graph is N = height * width, that is, the
    number of points in the grid.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Grid2d(shape=(32,))

    """

    def __init__(self, shape=(3,), **kwargs):
        # Parse shape
        try:
            h = shape[0]
            try:
                w = shape[1]
            except IndexError:
                w = h
        except TypeError:
            h = shape
            w = h

        # Filling up the weight matrix this way is faster than
        # looping through all the grid points:
        diag_1 = np.ones((h * w - 1,))
        diag_1[(w - 1)::w] = 0
        stride = w
        diag_2 = np.ones((h * w - stride,))
        W = sparse.diags(diagonals=[diag_1, diag_2],
                         offsets=[-1, -stride],
                         shape=(h * w, h * w),
                         format='csr',
                         dtype='float')
        W = utils.symmetrize(W, symmetrize_type='full')

        x = np.kron(np.ones((h, 1)), (np.arange(w) / float(w)).reshape(w, 1))
        y = np.kron(np.ones((w, 1)), np.arange(h) / float(h)).reshape(h * w, 1)
        y = np.sort(y, axis=0)[::-1]
        coords = np.concatenate((x, y), axis=1)

        self.h = h
        self.w = w
        plotting = {"limits": np.array([-1. / self.w, 1 + 1. / self.w,
                                        1. / self.h, 1 + 1. / self.h])}

        super(Grid2d, self).__init__(W=W, gtype='2d-grid', coords=coords,
                                     plotting=plotting, **kwargs)
