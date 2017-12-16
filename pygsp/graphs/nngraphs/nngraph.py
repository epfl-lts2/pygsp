# -*- coding: utf-8 -*-

import traceback

import numpy as np
from scipy import sparse, spatial

from pygsp import utils
from pygsp.graphs import Graph  # prevent circular import in Python < 3.5

_logger = utils.build_logger(__name__)


def _import_pfl():
    try:
        import pyflann as pfl
    except Exception:
        raise ImportError('Cannot import pyflann. Choose another nearest '
                          'neighbors method or try to install it with '
                          'pip (or conda) install pyflann (or pyflann3).')
    return pfl


class NNGraph(Graph):
    r"""Nearest-neighbor graph from given point cloud.

    Parameters
    ----------
    Xin : ndarray
        Input points, Should be an `N`-by-`d` matrix, where `N` is the number
        of nodes in the graph and `d` is the dimension of the feature space.
    NNtype : string, optional
        Type of nearest neighbor graph to create. The options are 'knn' for
        k-Nearest Neighbors or 'radius' for epsilon-Nearest Neighbors (default
        is 'knn').
    use_flann : bool, optional
        Use Fast Library for Approximate Nearest Neighbors (FLANN) or not.
        (default is False)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width parameter of the similarity kernel (default is 0.1)
    epsilon : float, optional
        Radius for the epsilon-neighborhood search (default is 0.01)
    gtype : string, optional
        The type of graph (default is 'nearest neighbors')
    plotting : dict, optional
        Dictionary of plotting parameters. See :obj:`pygsp.plotting`.
        (default is {})
    symmetrize_type : string, optional
        Type of symmetrization to use for the adjacency matrix. See
        :func:`pygsp.utils.symmetrization` for the options.
        (default is 'average')
    dist_type : string, optional
        Type of distance to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
        (default is 'euclidean')
    order : float, optional
        Only used if dist_type is 'minkowski'; represents the order of the
        Minkowski distance. (default is 0)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.RandomState(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(X)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, Xin, NNtype='knn', use_flann=False, center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01, gtype=None,
                 plotting={}, symmetrize_type='average', dist_type='euclidean',
                 order=0, **kwargs):

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
            bounding_radius = 0.5 * np.linalg.norm(np.amax(Xout, axis=0) -
                                                   np.amin(Xout, axis=0), 2)
            scale = np.power(N, 1. / float(min(d, 3))) / 10.
            Xout *= scale / bounding_radius

        # Translate distance type string to corresponding Minkowski order.
        dist_translation = {"euclidean": 2,
                            "manhattan": 1,
                            "max_dist": np.inf,
                            "minkowski": order
                            }

        if self.NNtype == 'knn':
            spi = np.zeros((N * k))
            spj = np.zeros((N * k))
            spv = np.zeros((N * k))

            if self.use_flann:
                pfl = _import_pfl()
                pfl.set_distance_type(dist_type, order=order)
                flann = pfl.FLANN()

                # Default FLANN parameters (I tried changing the algorithm and
                # testing performance on huge matrices, but the default one
                # seems to work best).
                NN, D = flann.nn(Xout, Xout, num_neighbors=(k + 1),
                                 algorithm='kdtree')

            else:
                kdt = spatial.KDTree(Xout)
                D, NN = kdt.query(Xout, k=(k + 1),
                                  p=dist_translation[dist_type])

            for i in range(N):
                spi[i * k:(i + 1) * k] = np.kron(np.ones((k)), i)
                spj[i * k:(i + 1) * k] = NN[i, 1:]
                spv[i * k:(i + 1) * k] = np.exp(-np.power(D[i, 1:], 2) /
                                                float(self.sigma))

        elif self.NNtype == 'radius':

            kdt = spatial.KDTree(Xout)
            D, NN = kdt.query(Xout, k=None, distance_upper_bound=epsilon,
                              p=dist_translation[dist_type])
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

        else:
            raise ValueError('Unknown NNtype {}'.format(self.NNtype))

        W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError('Weight matrix W is not square')

        # Enforce symmetry. Note that checking symmetry with
        # np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method=symmetrize_type)

        super(NNGraph, self).__init__(W=W, gtype=gtype, plotting=plotting,
                                      coords=Xout, **kwargs)
