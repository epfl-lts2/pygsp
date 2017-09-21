# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Minnesota(Graph):
    r"""Minnesota road network (from MatlabBGL).

    Parameters
    ----------
    connect : bool
        If True, the adjacency matrix is adjusted so that all edge weights are
        equal to 1, and the graph is connected. Set to False to get the
        original disconnected graph.

    References
    ----------
    See :cite:`gleich`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Minnesota()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, connect=True, **kwargs):

        data = utils.loadmat('pointclouds/minnesota')
        self.labels = data['labels']
        A = data['A']

        plotting = {"limits": np.array([-98, -89, 43, 50]),
                    "vertex_size": 40}

        if connect:

            # Missing edges needed to connect the graph.
            A = sparse.lil_matrix(A)
            A[348, 354] = 1
            A[354, 348] = 1
            A = sparse.csc_matrix(A)

            # Binarize: 8 entries are equal to 2 instead of 1.
            A = (A > 0).astype(bool)

            gtype = 'minnesota'

        else:

            gtype = 'minnesota-disconnected'

        super(Minnesota, self).__init__(W=A, coords=data['xy'], gtype=gtype,
                                        plotting=plotting, **kwargs)
