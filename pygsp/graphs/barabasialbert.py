# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class BarabasiAlbert(Graph):
    r"""Barabasi-Albert preferential attachment.

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
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.BarabasiAlbert(N=150, seed=42)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    """
    def __init__(self, N=1000, m0=1, m=1, seed=None, **kwargs):

        if m > m0:
            raise ValueError('Parameter m cannot be above parameter m0.')

        self.m0 = m0
        self.m = m
        self.seed = seed

        W = sparse.lil_matrix((N, N))
        rng = np.random.default_rng(seed)

        for i in range(m0, N):
            distr = W.sum(axis=1)
            distr += np.concatenate((np.ones((i, 1)), np.zeros((N-i, 1))))

            connections = rng.choice(
                N, size=m, replace=False, p=np.ravel(distr / distr.sum()))
            for elem in connections:
                W[elem, i] = 1
                W[i, elem] = 1

        super(BarabasiAlbert, self).__init__(W, **kwargs)

    def _get_extra_repr(self):
        return dict(m0=self.m0, m=self.m, seed=self.seed)
