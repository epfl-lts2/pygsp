#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

# Constants
LMAX_TOL = 1e-6


class LaplacianType:
    DEFAULT = 0
    NORMALIZED = 1


class Graph:
    """The Graph class is composed of two main components : a SpectralProp
     object and an AttributeMap object.

    """
    def __init__(self, W, laplacianType=LaplacianType.DEFAULT):
        self.prop = SpectralGraph(W, laplacianType)
        self.attrs = AttributeMap()


class SpectralGraph:
    """Container of all spectral infos of the graph"""
    def __init__(self, W, laplacianType):
        self.L = compute_laplacian(W, laplacianType)  # Laplacian matrix
        self.U = None  # eigenvectors
        self.E = None  # eigenvalue array
        self.lmax = None  # largest eigenvalue
        self.W = None  # placeholder for weight matrix if you need it

    def compute_eig_decomp(self):
        """Compute eigen decomposition of the laplacian and set
           U, E and lmax

        """
        self.E, self.U = compute_eigen_decomp(self.L)
        self.lmax = self.E[-1]

    def compute_lmax(self, tol=LMAX_TOL):
        self.lmax = compute_lmax(self.L, tol)[0]


class AttributeMap:
    """The AttributeMap class is used to store the labels
        and coordinates map of the graph.

    """
    def __init__(self, attrs=dict(dict())):
        self.attrs = attrs

    def load_from_file(self, path):
        """TODO"""
        pass


###### Module free functions ######


def compute_eigen_decomp(L):
    """Computes the eigenvalue decomposition of a symmetric laplacian
       matrix. The eigenvalues are sorted in ascending order.
    """
    tmp = None
    if type(L) is np.ndarray:
        tmp = L
    else:
        tmp = L.todense()

    # Compute eigenvalues for symmetric matrix
    E, U = np.linalg.eigh(tmp)
    # Reorder eigenvalues
    idx = np.argsort(E)  # sorting eigenvalues and eigenvectors
    E = E[idx]
    U = U[:, idx]
    return E, U


def compute_lmax(L, tol=LMAX_TOL):
    """Approximate the largest eigenvalue of a symmetric matrix"""
    return sp.sparse.linalg.eigsh(L, 1, tol=tol)


def compute_laplacian(W, laplacianType):
    """Computes the laplacian of a graph from its weight matrix
        Mostly inspired by:
        https://github.com/aweinstein/PySGWT/blob/master/sgwt.py

        Parameters
        ----------
        W : numpy array or scipy sparse matrix
        normalized : normalized laplacian

        Returns
        -------
        L : Sparse laplacian matrix (csr format)

    """

    # N = weightMatrix.shape[0]
    # # TODO: Raise exception if A is not square

    # degrees = weightMatrix.sum(1)
    # # To deal with loops, must extract diagonal part of A
    # diagw = np.diag(weightMatrix)

    # # w will consist of non-diagonal entries only
    # ni2, nj2 = weightMatrix.nonzero()
    # w2 = weightMatrix[ni2, nj2]
    # ndind = (ni2 != nj2).nonzero() # Non-diagonal indices
    # ni = ni2[ndind]
    # nj = nj2[ndind]
    # w = w2[ndind]

    # di = np.arange(N) # diagonal indices

    # if laplacian_type == 'raw':
    #     # non-normalized laplaciand L = D - A
    #     L = np.diag(degrees - diagw)
    #     L[ni, nj] = -w
    # elif laplacian_type == 'normalized':
    #     # TODO: Implement the normalized laplacian case
    #     # % normalized laplacian D^(-1/2)*(D-A)*D^(-1/2)
    #     # % diagonal entries
    #     # dL=(1-diagw./degrees); % will produce NaN for degrees==0 locations
    #     # dL(degrees==0)=0;% which will be fixed here
    #     # % nondiagonal entries
    #     # ndL=-w./vec( sqrt(degrees(ni).*degrees(nj)) );
    #     # L=sparse([ni;di],[nj;di],[ndL;dL],N,N);
    #     print 'Not implemented'
    # else:
    #     # TODO: Raise an exception
    #     print "Don't know what to do"
    return None
