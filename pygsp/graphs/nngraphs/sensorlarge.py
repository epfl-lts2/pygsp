# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class SensorLarge(NNGraph):
    r"""Random sensor graph.

    When creating large graphs, it is more computationally efficient than the
    Sensor graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 64)
    distribute : bool
        To distribute the points more evenly (default = False)
    seed : int
        Seed for the random number generator (for reproducible graphs).
    k : number of neighboors (Default 6)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SensorLarge(N=64, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, N=10000, distribute=False, seed=None, k=6, **kwargs):

        self.distribute = distribute
        self.seed = seed

        gtype = 'SensorLarge'

        plotting = {'limits': np.array([0, 1, 0, 1])}

        coords = self._create_coords(N, distribute)

        super(SensorLarge, self).__init__(Xin=coords, k=k, gtype=gtype,
                                          plotting=plotting, **kwargs)


    def _create_coords(self, N, distribute):
        XCoords = np.zeros((N, 1))
        YCoords = np.zeros((N, 1))

        rs = np.random.RandomState(self.seed)

        if distribute:
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

        return coords
