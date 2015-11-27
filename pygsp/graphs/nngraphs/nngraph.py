# -*- coding: utf-8 -*-

from .. import Graph
from pygsp.graphs import gutils

import numpy as np
from scipy import sparse, spatial


class NNGraph(Graph):
    r"""
    Creates a graph from a pointcloud.

    Parameters
    ----------
    Xin : ndarray
        Input points
    use_flann : bool
        Whether flann method should be used (knn is otherwise used).
        (default is False)
        (this option is not implemented yet)
    center : bool
        Center the data (default is True)
    rescale : bool
        Rescale the data (in a 1-ball) (default is True)
    k : int
        Number of neighbors for knn (default is 10)
    sigma : float
        Variance of the distance kernel (default is 0.1)
    epsilon : float
        RRdius for the range search (default is 0.01)
    gtype : string
        The type of graph (default is "knn")

    Examples
    --------
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> Xin = np.arange(90).reshape(30, 3)
    >>> G = graphs.NNGraph(Xin)

    """

    def __init__(self, Xin, NNtype='knn', use_flann=False, center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01, gtype=None,
                 plotting={}, symmetrize_type='average', **kwargs):

        if Xin is None:
            raise ValueError('You must enter a Xin to process the NNgraph')
        else:
            self.Xin = Xin

        self.NNtype = NNtype
        self.use_flann = use_flann
        self.center = center
        self.rescale = rescale
        self.k = k
        self.sigma = sigma
        self.epsilon = epsilon

        if gtype is None:
            gtype = 'nearest neighbors'
        else:
            gtype = '{}, NNGraph'.format(gtype)

        self.symmetrize_type = symmetrize_type

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if self.center:
            Xout = self.Xin - np.kron(np.ones((N, 1)),
                                      np.mean(self.Xin, axis=0))

        if self.rescale:
            bounding_radius = 0.5*np.linalg.norm(np.amax(Xout, axis=0) -
                                                 np.amin(Xout, axis=0), 2)
            scale = np.power(N, 1./float(min(d, 3)))/10.
            Xout *= scale / bounding_radius

        if self.NNtype == 'knn':
            spi = np.zeros((N*k))
            spj = np.zeros((N*k))
            spv = np.zeros((N*k))

            # since we didn't find a good flann python library yet, we wont implement it for now
            if self.use_flann:
                raise NotImplementedError('Suitable library for flann has not '
                                          'been found yet.')
            else:
                kdt = spatial.KDTree(Xout)
                D, NN = kdt.query(Xout, k=k + 1)

            for i in range(N):
                spi[i*k:(i + 1)*k] = np.kron(np.ones((k)), i)
                spj[i*k:(i + 1)*k] = NN[i, 1:]
                spv[i*k:(i + 1)*k] = np.exp(-np.power(D[i, 1:], 2) /
                                            float(self.sigma))

            W = sparse.csc_matrix((spv, (spi, spj)),
                                  shape=(np.shape(self.Xin)[0],
                                         np.shape(self.Xin)[0]))

        elif self.NNtype == 'radius':

            kdt = spatial.KDTree(Xout)
            D, NN = kdt.query(Xout, k=None, distance_upper_bound=epsilon)
            count = 0
            for i in range(N):
                count = count + len(NN[i])

            spi = np.zeros((count))
            spj = np.zeros((count))
            spv = np.zeros((count))

            start = 0
            for i in range(N):
                leng = len(NN[i]) - 1
                spi[start:start + leng] = np.kron(np.ones((leng)), i)
                spj[start:start + leng] = NN[i][1:]
                spv[start:start + leng] = np.exp(-np.power(D[i][1:], 2) /
                                                 float(self.sigma))
                start = start + leng

            W = sparse.csc_matrix((spv, (spi, spj)),
                                  shape=(np.shape(self.Xin)[0],
                                         np.shape(self.Xin)[0]))

        else:
            raise ValueError('Unknown type : allowed values are knn, radius')

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError('Weight matrix W is not square')

        # Symmetry checks
        if np.abs(W - W.T).sum():
            if symmetrize_type == 'average':
                W = (W + W.T) / 2.

            elif symmetrize_type == 'full':
                A = W > 0
                M = (A - (A.T * A))
                W = sparse.csr_matrix(W.T)
                W[M.T] = W.T[M.T]

            else:
                raise ValueError("Unknown symmetrize type.")
        else:
            pass

        super(NNGraph, self).__init__(W=W, gtype=gtype, plotting=plotting,
                                      coords=Xout, **kwargs)
