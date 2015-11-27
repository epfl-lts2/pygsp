# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger

import numpy as np
from scipy import sparse
import random as rd
from math import floor


class RandomRegular(Graph):
    r"""
    Create a random regular graphs

    The random regular graph has the property that every nodes is connected to
    'k' other nodes.

    Parameters
    ----------
    N : int
        Number of nodes (default is 64)
    k : int
        Number of connections of each nodes (default is 6)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.RandomRegular()

    """

    def isRegularGraph(self, A):
        r"""
        Troubleshoot a given regular graph.

        Parameters
        ----------
        A : sparse matrix

        """
        warn = False
        msg = 'The given matrix'

        # check if the sparse matrix is in a good format
        if A.getformat() == 'lil' or \
                A.getformat() == 'dia' or \
                A.getformat() == 'bok':
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

    def createRandRegGraph(self, vertNum, deg, maxIter=10):
        r"""
        Create a simple d-regular undirected graph.

        simple = without loops or double edges
        d-reglar = each vertex is adjecent to d edges

        Parameters
        ----------
        vertNum : int
            Number of vertices
        deg : int
            The degree of each vertex
        maxIter : int
            The maximum number of iterations

        Returns
        -------
        A : sparse
            Representation of the graph

        Algorithm
        ---------
        "The pairing model": create n*d 'half edges'.
        Repeat as long as possible: pick a pair of half edges
        and if it's legal (doesn't create a loop nor a double edge)
        add it to the graph

        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.7957&rep=rep1&type=pdf
        (This code has been adapted from matlab to python)

        """
        n = vertNum
        d = deg

        # continue until a proper graph is formed
        if (n * d) % 2 == 1:
            raise ValueError("createRandRegGraph input err:\
                                n*d must be even!")

        # a list of open half-edges
        U = np.kron(np.ones((d)), np.arange(n))

        # the graphs adjacency matrix
        A = sparse.lil_matrix(np.zeros((n, n)))

        edgesTested = 0
        repetition = 1

        # check that there are no loops nor parallel edges
        while np.size(U) and repetition < matIter:
            edgesTested += 1

            # print(progess)
            if edgesTested % 5000 == 0:
                self.logger.debug("createRandRegGraph() progress: edges= "
                                  "{}/{}.".format(edgesTested, n*d/2))

            # chose at random 2 half edges
            i1 = floor(rd.random()*np.shape(U)[0])
            i2 = floor(rd.random()*np.shape(U)[0])
            v1 = U[i1]
            v2 = U[i2]

            # check that there are no loops nor parallel edges
            if v1 == v2 or A[v1, v2] == 1:
                # restart process if needed
                if edgesTested == n*d:
                    repetition = repetition + 1
                    edgesTested = 0
                    U = np.kron(np.ones((d)), np.arange(n))
                    A = sparse.lil_matrix(np.zeros((n, n)))
            else:
                # add edge to graph
                A[v1, v2] = 1
                A[v2, v1] = 1

                # remove used half-edges
                v = sorted([i1, i2])
                U = np.concatenate((U[:v[0]], U[v[0] + 1:v[1]], U[v[1] + 1:]))

        self.isRegularGraph(A)

        return A

    def __init__(self, N=64, k=6, **kwargs):
        self.k = k

        # Build the logger as createRandRegGraph need it
        self.logger = build_logger(__name__, **kwargs)

        W = self.createRandRegGraph(N, k)

        super(RandomRegular, self).__init__(W=W, gtype="random_regular",
                                            **kwargs)
