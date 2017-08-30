# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class RandomRegular(Graph):
    r"""Random k-regular graph.

    The random regular graph has the property that every node is connected to
    k other nodes. That graph is simple (without loops or double edges),
    k-regular (each vertex is adjacent to k nodes), and undirected.

    Parameters
    ----------
    N : int
        Number of nodes (default is 64)
    k : int
        Number of connections, or degree, of each node (default is 6)
    maxIter : int
        Maximum number of iterations (default is 10)

    Notes
    -----
    The *pairing model* algorithm works as follows. First create n*d *half
    edges*. Then repeat as long as possible: pick a pair of half edges and if
    it's legal (doesn't create a loop nor a double edge) add it to the graph.

    References
    ----------
    See :cite:`kim2003randomregulargraphs`.
    This code has been adapted from matlab to python.

    Examples
    --------
    >>> G = graphs.RandomRegular()

    """

    def __init__(self, N=64, k=6, maxIter=10, **kwargs):
        self.k = k

        self.logger = utils.build_logger(__name__, **kwargs)

        # continue until a proper graph is formed
        if (N * k) % 2 == 1:
            raise ValueError("input error: N*d must be even!")

        # a list of open half-edges
        U = np.kron(np.ones(k), np.arange(N))

        # the graphs adjacency matrix
        A = sparse.lil_matrix(np.zeros((N, N)))

        edgesTested = 0
        repetition = 1

        # check that there are no loops nor parallel edges
        while np.size(U) and repetition < maxIter:
            edgesTested += 1

            # print(progess)
            if edgesTested % 5000 == 0:
                self.logger.debug("createRandRegGraph() progress: edges= "
                                  "{}/{}.".format(edgesTested, N*k/2))

            # chose at random 2 half edges
            i1 = np.random.randint(0, np.shape(U)[0])
            i2 = np.random.randint(0, np.shape(U)[0])
            v1 = U[i1]
            v2 = U[i2]

            # check that there are no loops nor parallel edges
            if v1 == v2 or A[v1, v2] == 1:
                # restart process if needed
                if edgesTested == N*k:
                    repetition = repetition + 1
                    edgesTested = 0
                    U = np.kron(np.ones(k), np.arange(N))
                    A = sparse.lil_matrix(np.zeros((N, N)))
            else:
                # add edge to graph
                A[v1, v2] = 1
                A[v2, v1] = 1

                # remove used half-edges
                v = sorted([i1, i2])
                U = np.concatenate((U[:v[0]], U[v[0] + 1:v[1]], U[v[1] + 1:]))

        super(RandomRegular, self).__init__(W=A, gtype="random_regular",
                                            **kwargs)

        self.is_regular()

    def is_regular(self):
        r"""
        Troubleshoot a given regular graph.

        """
        warn = False
        msg = 'The given matrix'

        # check if the sparse matrix is in a good format
        A = self.A
        if A.getformat() in ['lil', 'dia', 'bok']:
            A = A.tocsc()

        # check symmetry
        tmp = A - A.T
        if np.abs(tmp).sum() > 0:
            warn = True
            msg = '{} is not symmetric,'.format(msg)

        # check parallel edged
        if A.max(axis=None) > 1:
            warn = True
            msg = '{} has parallel edges,'.format(msg)

        # check that d is d-regular
        d_vec = A.sum(axis=0)
        if np.min(d_vec) != np.max(d_vec):
            warn = True
            msg = '{} is not d-regular,'.format(msg)

        # check that g doesn't contain any self-loop
        if A.diagonal().any():
            warn = True
            msg = '{} has self loop.'.format(msg)

        if warn:
            self.logger.warning('{}.'.format(msg[:-1]))
        else:
            self.logger.info('{} is ok.'.format(msg))
