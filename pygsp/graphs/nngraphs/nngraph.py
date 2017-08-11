# -*- coding: utf-8 -*-

import numpy as np

from .. import Graph
from ...utils import symmetrize
from scipy import sparse, spatial

try:
    import pyflann as fl
    pfl_import = True
except Exception as e:
    print('ERROR : Could not import pyflann. Try to install it for faster kNN computations.')
    pfl_import = False


class NNGraph(Graph):
    r"""
    Creates a graph from a pointcloud.

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
    >>> from pygsp import graphs
    >>> import numpy as np
    >>> Xin = np.arange(90).reshape(30, 3)
    >>> G = graphs.NNGraph(Xin)

    """

    def __init__(self, Xin, NNtype='knn', use_flann=False, center=True,
                 rescale=True, k=10, sigma=0.1, epsilon=0.01, gtype=None,
                 plotting={}, symmetrize_type='average', dist_type='euclidean',
                 order=0, **kwargs):

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

            if self.use_flann and pfl_import:
                fl.set_distance_type(dist_type, order=order)
                flann = fl.FLANN()

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
            raise ValueError('Unknown type : allowed values are knn, radius')

        """
        Before, we were calling this same snippet in each of the conditional
        statements above, so it's better to call it only once, after the
        conditional statements have been evaluated.
        """
        W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError('Weight matrix W is not square')

        # Enforce symmetry
        """
        This conditional statement costs the same amount of computation as the
        symmetrization itself, so it's better to simply call utils.symmetrize
        no matter what
        if np.abs(W - W.T).sum():
            W = utils.symmetrize(W, symmetrize_type=symmetrize_type)
        else:
            pass
        """
        W = symmetrize(W, symmetrize_type=symmetrize_type)

        super(NNGraph, self).__init__(W=W, gtype=gtype, plotting=plotting,
                                      coords=Xout, **kwargs)
