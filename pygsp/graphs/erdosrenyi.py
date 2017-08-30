# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class ErdosRenyi(Graph):
    r"""Erdos Renyi graph.

    The Erdos Renyi graph is constructed by connecting nodes randomly. Each
    edge is included in the graph with probability p independent from every
    other edge. All edge weights are equal to 1.

    Parameters
    ----------
    N : int
        Number of nodes (default is 100)
    p : float
        Probability of connection of a node with another
    connected : bool
        Force the graph to be connected (default is False).
    directed : bool
        Define if the graph is directed (default is False).
    max_iter : int
        Maximum number of try to connect the graph (default is 10).

    Examples
    --------
    >>> G = graphs.ErdosRenyi(100, 0.05)

    """

    def __init__(self, N=100, p=0.1, connected=False, directed=False,
                 max_iter=10, **kwargs):
        self.p = p

        if not 0 < p < 1:
            raise ValueError('Probability p should be in [0, 1].')

        M = int(N * (N-1) if directed else N * (N-1) / 2)
        nb_elem = int(p * M)

        for i in range(max_iter):
            indices = np.random.permutation(M)[:nb_elem]

            if directed:
                all_ind = np.tril_indices(N, N-1)
                non_diag = tuple(map(lambda dim: dim[condlist], all_ind))
                indices = tuple(map(lambda coord: coord[indices], non_diag))
            else:
                indices = tuple(map(lambda coord: coord[indices], np.tril_indices(N, -1)))

            matrix = sparse.csr_matrix((np.ones(nb_elem), indices), shape=(N, N))
            self.W = matrix if directed else matrix + matrix.T
            self.A = self.W > 0

            if not connected or self.is_connected(recompute=True):
                break

        super(ErdosRenyi, self).__init__(W=self.W, gtype=u"Erdös Renyi", **kwargs)
