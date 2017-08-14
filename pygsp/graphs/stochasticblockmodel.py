# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph


class StochasticBlockModel(Graph):
    r"""
    Create a graph generated with the Stochastic Block Model.

    The Stochastic Block Model graph is constructed by connecting nodes with a
    probability which depends on the cluster of the two nodes.  One can define
    the clustering association of each node, denoted by vector z, but also the
    probability matrix M.  All edge weights are equal to 1. By default, Mii >
    Mjk and nodes are uniformly clusterized.

    Parameters
    ----------
    N : int
        Number of nodes (default is 1024)
    k : float
        Number of classes (default is 5)
    z : array_like
        the vector of length N containing the association between nodes and
        classes. Default uniform.
    M : array_like
        the k by k matrix containing the probability of connecting nodes based
        on their class belonging. Default using p and q.
    p : float or array_like
        the diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a length k vector. Default p = 0.7.
    q : float or array_like
        the off-diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a k x k matrix, diagonal will be
        discarded. Default q = 0.3/k.
    undirected : bool
        force the graph to be undirected. Default True.
    no_self_loop : bool
        force the graph to have no self loop. Default True.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.StochasticBlockModel(N=1024, k=5)

    """

    def __init__(self, N=1024, k=5, z=None, M=None, p=0.7, q=None,
                 undirected=True, no_self_loop=True, **kwargs):

        if z is None:
            z = np.random.randint(0, k, N)

        if M is None:
            if isinstance(p, float):
                p *= np.ones(k)
            elif isinstance(p, list):
                p = np.array(p)

            if p.shape != (k, ):
                raise ValueError('Optional parameter p is neither a scalar nor'
                                 'a vector of length k.')

            if q is None:
                q = 0.3 / k
            if isinstance(q, float):
                q *= np.ones((k, k))
            elif isinstance(q, list):
                q = np.array(q)

            if q.shape != (k, k):
                raise ValueError('Optional parameter q is neither a scalar nor'
                                 'a matrix of size k x k.')

            M = q
            M.flat[::k+1] = p  # edit the diagonal terms

        nb_row, nb_col = 0, 0
        csr_data, csr_i, csr_j = [], [], []
        for _ in range(N**2):
            if nb_row != nb_col or not no_self_loop:
                if nb_row > nb_col or not undirected:
                    if np.random.rand() < M[z[nb_row], z[nb_col]]:
                        csr_data.append(1)
                        csr_i.append(nb_row)
                        csr_j.append(nb_col)
            if nb_row < N-1:
                nb_row += 1
            else:
                nb_row = 0
                nb_col += 1

        W = sparse.csr_matrix((csr_data, (csr_i, csr_j)), shape=(N, N))

        if undirected:
            W = W + W.T

            if not no_self_loop:
                # avoid doubling the self loops with the above sum
                W[np.arange(N), np.arange(N)] /= 2.

        self.info = {'node_com': z, 'comm_sizes': np.bincount(z),
                     'world_rad': np.sqrt(N)}

        gtype = 'StochasticBlockModel'
        super(StochasticBlockModel, self).__init__(gtype=gtype, W=W, **kwargs)
