# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger

import numpy as np
from scipy import sparse
import random as rd
from math import floor


class BarabasiAlbert(Graph):
    r"""
    Create a random Barabasi-Albert graph.

    The Barabasi-Albert graph is constructed by connecting nodes in two steps.
    First, m0 nodes are connected at random. Then, nodes are added one by one.
    Each node is connected to m of the older nodes with a probability distribution
    depending of the node-degrees of the other nodes:
    ::
        p_n(i) = \frac{k_i}{\sum_j{k_j}}
    ::

    For the moment, we set m0 = m = 1.

    Parameters
    ----------
    N : int
        Number of nodes (default is 1000)
    m0 : int
        Number of initial nodes (default is 1)
    m : int
        Number of connections at each step (default is 1)
        m can never be larger than m0.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.BarabasiAlbert(500)

    """
    def __init__(self, N=1000, m0=1, m=1, **kwargs):
        if m > m0:
            raise ValueError("GSP_BarabasiAlbert: The parameter m "
                             "cannot be above m0.")

        W = sparse.lil_matrix((N, N))

        if m0 > 1:
            raise NotImplementedError("Initial connection of the nodes is not "
                                      "implemented yet. Please keep m0 = 1.")

        for i in range(m0, N):
            distr = W.sum(axis=1)
            if distr.sum() == 0:
                W[0, 1] = 1
                W[1, 0] = 1
            else:
                connections = np.random.choice(N, size=m, replace=False, p=np.ravel(distr/distr.sum()))
                for elem in connections:
                    W[elem, i] = 1
                    W[i, elem] = 1

        super(BarabasiAlbert, self).__init__(W=W, gtype=u"Barabási-Albert", **kwargs)
