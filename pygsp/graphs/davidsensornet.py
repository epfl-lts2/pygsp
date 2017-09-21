# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class DavidSensorNet(Graph):
    r"""Sensor network.

    Parameters
    ----------
    N : int
        Number of vertices (default = 64). Values of 64 and 500 yield
        pre-computed and saved graphs. Other values yield randomly generated
        graphs.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.DavidSensorNet()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N=64, seed=None, **kwargs):
        if N == 64:
            data = utils.loadmat('pointclouds/david64')
            assert data['N'][0, 0] == N
            W = data['W']
            coords = data['coords']

        elif N == 500:
            data = utils.loadmat('pointclouds/david500')
            assert data['N'][0, 0] == N
            W = data['W']
            coords = data['coords']

        else:
            coords = np.random.RandomState(seed).rand(N, 2)

            target_dist_cutoff = -0.125 * N / 436.075 + 0.2183
            T = 0.6
            s = np.sqrt(-target_dist_cutoff**2/(2*np.log(T)))
            d = utils.distanz(coords.T)
            W = np.exp(-np.power(d, 2)/(2.*s**2))
            W[W < T] = 0
            W[np.diag_indices(N)] = 0

        plotting = {"limits": [0, 1, 0, 1]}

        super(DavidSensorNet, self).__init__(W=W, gtype='davidsensornet',
                                             coords=coords, plotting=plotting,
                                             **kwargs)
