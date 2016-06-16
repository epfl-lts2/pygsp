# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger

import numpy as np
from scipy import sparse
import random as rd
from math import floor


class ErdosRenyi(Graph):
    r"""
    Create a random Erdos Renyi graph.

    The Erdos Renyi graph is constructed by connecting nodes randomly. Each 
    edge is included in the graph with probability p independent from every
    other edge. All edge weights are equal to 1.

    Parameters
    ----------
    N : int
        Number of nodes (default is 100)
    p : float
        Probability of connection of a node with another
    param :
        Structure of optional parameter
        connected - flag to force the graph to be connected. By default, it is False.
        directed - define if the graph is directed. By default, it is False.
        max_iter - is the maximum number of try to connect the graph. By default, it is 10.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.ErdosRenyi(100, 0.05)

    """

    def __init__(self, N=100, p=0.1, **kwargs):
        self.p = p

        need_connected = bool(kwargs.pop('connected', False))
        directed = bool(kwargs.pop('directed', False))
        max_iter = int(kwargs.pop('max_iter', 10))

        if p > 1:
            raise ValueError("GSP_ErdosRenyi: The probability p "
                             "cannot be above 1.")
        elif p < 0:
            raise ValueError("GSP_ErdosRenyi: The probability p "
                             "cannot be negative.")

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
            is_connected = self.is_connected()

            if not need_connected or is_connected:
                break

        super(ErdosRenyi, self).__init__(W=self.W, gtype=u"Erdös Renyi", **kwargs)
