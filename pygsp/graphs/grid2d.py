# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from . import Graph
from .. import utils


class Grid2d(Graph):
    r"""
    Create a 2-dimensional grid graph.

    Parameters
    ----------
    shape : tuple
        Dimensions of the 2-dimensional grid. Syntax: (height, width), or
        (height,), in which case one has width = height.

    Notes
    -----
    The total number of nodes on the graph is N = height * width, that is, the
    number of point in the grid.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Grid2d(shape=(32,))

    """

    def __init__(self, shape=(3,), **kwargs):
        # (Rodrigo) I think using a single shape parameter, and calling the
        # dimensions of the grid 'height' (h) and 'width' (w) make more sense
        # than the previous Nv and Mv.
        try:
            h, w = shape
        except ValueError:
            h = shape[0]
            w = h

        # Filling up the weight matrix this way is faster than looping through
        # all the grid points:
        diag_1 = np.ones((h * w - 1,))
        diag_1[(w - 1)::w] = 0
        diag_3 = np.ones((h * w - 3,))
        W = sparse.diags(diagonals=[diag_1, diag_3],
                         offsets=[-1, -3],
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
        plotting = {"vertex_size": 30,
                    "limits": np.array([-1. / self.w, 1 + 1. / self.w,
                                        1. / self.h, 1 + 1. / self.h])}

        super(Grid2d, self).__init__(W=W, gtype='2d-grid', coords=coords,
                                     plotting=plotting, **kwargs)
