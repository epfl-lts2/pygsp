# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger
from pygsp.graphs.gutils import check_connectivity

import numpy as np
from scipy import sparse
import random as rd
from math import floor


class ErdosRenyi(Graph):
    r"""
    Create a random Erdos Renyi graph

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
        max_iter - is the maximum number of try to connect the graph. By default, it is 10.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.ErdosRenyi(100, 0.05)

    Author: Lionel Martin
    """

    def __init__(self, N=100, p=0.1, **kwargs):
        self.N = N
        self.p = p

        connected = bool(kwargs.pop('connected', False))
        max_iter = int(kwargs.pop('max_iter', 10))

        self.gtype = u"Erdös Renyi"
        self.logger = build_logger(__name__, **kwargs)

        if p > 1:
            raise ValueError("GSP_ErdosRenyi: The probability p \
                              cannot be above 1.")
        elif p < 0:
            raise ValueError("GSP_ErdosRenyi: The probability p \
                              cannot be negative.")

        is_connected = False
        M = self.N * (self.N-1) / 2
        nb_elem = int(self.p * M)

        for i in range(max_iter):
            if is_connected:
                break

            indices = np.random.permutation(M)[:nb_elem]
            indices = tuple(map(lambda coord: coord[indices], np.tril_indices(self.N, -1)))
            matrix = sparse.csr_matrix((np.ones(nb_elem), indices), shape=(self.N, self.N))
            self.W = matrix + matrix.T
            is_connected = check_connectivity(self)

        super(ErdosRenyi, self).__init__(W=self.W, gtype=self.gtype,
                                         **kwargs)
