# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class NNSensor(NNGraph):
    r"""Random sensor graph based on a NNgraph.

    When creating large graphs, it is more computationally efficient than the
    Sensor graph. Nevertheless, it has also less options.

    Parameters
    ----------
    N : int
        Number of nodes (default = 64)
    distribute : bool
        To distribute the points more evenly (default = False)
    seed : int
        Seed for the random number generator (for reproducible graphs).
    k : number of neighboors (Default 6)
    **kwargs : keyword arguments for NNGraph

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.NNSensor(N=64, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, N=10000, distributed=False, seed=None, k=6, **kwargs):
        """Initialize a Sentor graph based on NNgraph."""
        self.distributed = distributed
        self.seed = seed
        self.k = k

        plotting = {'limits': np.array([0, 1, 0, 1])}

        coords = self._create_coords(N, distributed)

        super(NNSensor, self).__init__(Xin=coords, k=k,
                                       plotting=plotting, **kwargs)

    def _create_coords(self, N, distributed):
        XCoords = np.zeros((N, 1))
        YCoords = np.zeros((N, 1))

        rs = np.random.RandomState(self.seed)

        if distributed:
            mdim = int(np.ceil(np.sqrt(N)))

            # for i in range(mdim):
            #     for j in range(mdim):
            #         if i*mdim + j < N:
            #             XCoords[i*mdim + j] = np.array((i + rs.rand()) / mdim)
            #             YCoords[i*mdim + j] = np.array((j + rs.rand()) / mdim)
            XCoords = np.repeat(np.arange(mdim),mdim) + rs.rand(mdim**2)
            YCoords = np.tile(np.arange(mdim), mdim) + rs.rand(mdim**2)

            XCoords /= mdim
            YCoords /= mdim
            XCoords = XCoords[:N].reshape((N, 1))
            YCoords = YCoords[:N].reshape((N, 1))
        # take random coordinates in a 1 by 1 square
        else:
            XCoords = rs.rand(N, 1)
            YCoords = rs.rand(N, 1)

        coords = np.concatenate((XCoords, YCoords), axis=1)

        return coords

    def _get_extra_repr(self):
        return {'k': self.k,
                'distributed': self.distributed,
                'seed': self.seed}
