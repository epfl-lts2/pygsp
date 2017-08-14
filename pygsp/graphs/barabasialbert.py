# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger

import numpy as np
from scipy import sparse


class BarabasiAlbert(Graph):
    r"""
    Create a graph following the preferential attachment concept like Barabasi-Albert graphs.

    The Barabasi-Albert graph is constructed by connecting nodes in two steps.
    First, m0 nodes are created. Then, nodes are added one by one.

    By lack of clarity, we take the liberty to create it as follows:

        1. the m0 initial nodes are disconnected,
        2. each node is connected to m of the older nodes with a probability
           distribution depending of the node-degrees of the other nodes,
           :math:`p_n(i) = \frac{1 + k_i}{\sum_j{1 + k_j}}`.

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

        for i in range(m0, N):
            distr = W.sum(axis=1)
            distr += np.concatenate((np.ones((i, 1)), np.zeros((N-i, 1))))

            connections = np.random.choice(N, size=m, replace=False, p=np.ravel(distr/distr.sum()))
            for elem in connections:
                W[elem, i] = 1
                W[i, elem] = 1

        super(BarabasiAlbert, self).__init__(W=W, gtype=u"Barabasi-Albert", **kwargs)
