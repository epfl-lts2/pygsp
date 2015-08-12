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
        This fonction prints a message describing the problem of a given
        sparse matrix.

        Parameters
        ----------
        A : sparse matrix

        """

        msg = "The given matrix "

        # check if the sparse matrix is in a good format
        if A.getformat() == 'lil' or \
                A.getformat() == 'dia' or \
                A.getformat() == 'bok':
            A = A.tocsc()

        # check symmetry
        tmp = (A - A.getH())
        if np.sum((tmp.getH()*tmp).diagonal()) > 0:
            msg += "is not symetric, "

        # check parallel edged
        if A.max(axis=None) > 1:
            msg += "has parallel edges, "

        # check that d is d-regular
        d_vec = A.sum(axis=0)
        if np.min(d_vec) < d_vec[:, 0] and np.max(d_vec) > d_vec[:, 0]:
            msg += "not d-regular, "

        # check that g doesn't contain any loops
        if A.diagonal().any() > 0:
            msg += "has self loops, "

        else:
            msg += "is ok"
        self.logger.info(msg)

    def createRandRegGraph(self, vertNum, deg):
        r"""
        Create a simple d-regular undirected graph
        simple = without loops or double edges
        d-reglar = each vertex is adjecent to d edges

        Parameters
        ----------
        vertNum : int
            Number of vertices
        deg : int
            The degree of each vertex

        Returns
        -------
        A : sparse
            Representation of the graph

        Algorithm
        ---------
        "The pairing model": create n*d 'half edges'.
        Repeat as long as possible: pick a pair of half edges
        and if it's legal (doesn't creat a loop nor a double edge)
        add it to the graph

        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.7957&rep=rep1&type=pdf
        (This code has been adapted from matlab to python)
        """

        n = vertNum
        d = deg
        matIter = 10

        # continue until a proper graph is formed
        if (n*d) % 2 == 1:
            raise ValueError("createRandRegGraph input err:\
                                n*d must be even!")

        # a list of open half-edges
        U = np.kron(np.ones((d)), np.arange(n))

        # the graphs adajency matrix
        A = sparse.lil_matrix(np.zeros((n, n)))

        edgesTested = 0
        repetition = 1

        # check that there are no loops nor parallel edges
        while np.size(U) != 0 and repetition < matIter:
            edgesTested += 1

            # print progess
            if edgesTested % 5000 == 0:
                self.logger.debug("createRandRegGraph() progress: edges= "
                                  "{}/{}n".format(edgesTested, n*d))

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
                U = np.concatenate((U[1:v[0]], U[v[0] + 1:v[1]], U[v[1] + 1:]))

        self.isRegularGraph(A)

        return A

    def __init__(self, N=64, k=6, **kwargs):
        self.N = N
        self.k = k

        self.gtype = "random_regular"

        self.logger = build_logger(__name__)  # Build the logger as createRandRegGraph needit

        self.W = self.createRandRegGraph(self.N, self.k)

        super(RandomRegular, self).__init__(W=self.W, gtype=self.gtype,
                                            **kwargs)
