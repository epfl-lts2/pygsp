# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Sensor(Graph):
    r"""Random sensor graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 64)
    Nc : int
        Minimum number of connections (default = 2)
    regular : bool
        Flag to fix the number of connections to nc (default = False)
    n_try : int
        Number of attempt to create the graph (default = 50)
    distribute : bool
        To distribute the points more evenly (default = False)
    connected : bool
        To force the graph to be connected (default = True)
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sensor(N=64, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N=64, Nc=2, regular=False, n_try=50,
                 distribute=False, connected=True, seed=None, **kwargs):

        self.Nc = Nc
        self.regular = regular
        self.n_try = n_try
        self.distribute = distribute
        self.seed = seed

        self.logger = utils.build_logger(__name__)

        if connected:
            for x in range(self.n_try):
                W, coords = self._create_weight_matrix(N, distribute,
                                                       regular, Nc)
                self.W = W

                if self.is_connected(recompute=True):
                    break

                elif x == self.n_try - 1:
                    self.logger.warning('Graph is not connected.')
        else:
            W, coords = self._create_weight_matrix(N, distribute, regular, Nc)

        W = sparse.lil_matrix(W)
        W = utils.symmetrize(W, method='average')

        gtype = 'regular sensor' if self.regular else 'sensor'

        plotting = {'limits': np.array([0, 1, 0, 1])}

        super(Sensor, self).__init__(W=W, coords=coords, gtype=gtype,
                                     plotting=plotting, **kwargs)

    def _get_nc_connection(self, W, param_nc):
        Wtmp = W
        W = np.zeros(np.shape(W))

        for i in range(np.shape(W)[0]):
            l = Wtmp[i]
            for j in range(param_nc):
                val = np.max(l)
                ind = np.argmax(l)
                W[i, ind] = val
                l[ind] = 0

        W = utils.symmetrize(W, method='average')

        return W

    def _create_weight_matrix(self, N, param_distribute, regular, param_Nc):
        XCoords = np.zeros((N, 1))
        YCoords = np.zeros((N, 1))

        rs = np.random.RandomState(self.seed)

        if param_distribute:
            mdim = int(np.ceil(np.sqrt(N)))
            for i in range(mdim):
                for j in range(mdim):
                    if i*mdim + j < N:
                        XCoords[i*mdim + j] = np.array((i + rs.rand()) / mdim)
                        YCoords[i*mdim + j] = np.array((j + rs.rand()) / mdim)

        # take random coordinates in a 1 by 1 square
        else:
            XCoords = rs.rand(N, 1)
            YCoords = rs.rand(N, 1)

        coords = np.concatenate((XCoords, YCoords), axis=1)

        # Compute the distanz between all the points
        target_dist_cutoff = 2*N**(-0.5)
        T = 0.6
        s = np.sqrt(-target_dist_cutoff**2/(2*np.log(T)))
        d = utils.distanz(x=coords.T)
        W = np.exp(-d**2/(2.*s**2))
        W -= np.diag(np.diag(W))

        if regular:
            W = self._get_nc_connection(W, param_Nc)

        else:
            W2 = self._get_nc_connection(W, param_Nc)
            W = np.where(W < T, 0, W)
            W = np.where(W2 > 0, W2, W)

        W = sparse.csc_matrix(W)
        return W, coords
