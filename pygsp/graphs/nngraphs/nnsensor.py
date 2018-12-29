# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class NNSensor(NNGraph):
    r"""Random sensor graph based on a NNgraph.

    When creating large graphs, it is more computationally efficient than the
    Sensor graph. Nevertheless, it has also less options.

    Parameters
    ----------
    N : int
        Number of nodes.
        Must be a perfect square if ``distributed=True``.
    k : int
        Minimum number of neighbors.
    distributed : bool
        Whether to distribute the vertices more evenly on the plane.
        If False, coordinates are taken uniformly at random in a [0, 1] square.
        If True, the vertices are arranged on a perturbed grid.
    seed : int
        Seed for the random number generator (for reproducible graphs).
    **kwargs :
        Additional keyword arguments for :class:`NNGraph`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.NNSensor(N=64, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.NNSensor(N=64, distributed=True, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, N=64, k=6, distributed=False, seed=None, **kwargs):

        self.distributed = distributed
        self.seed = seed

        plotting = {'limits': np.array([0, 1, 0, 1])}

        rs = np.random.RandomState(self.seed)

        if distributed:

            m = np.sqrt(N)
            if not m.is_integer():
                raise ValueError('The number of vertices must be a '
                                 'perfect square if they are to be '
                                 'distributed on a grid.')

            coords = np.mgrid[0:1:1/m, 0:1:1/m].reshape(2, -1).T
            coords += rs.uniform(0, 1/m, (N, 2))

        else:

            coords = rs.uniform(0, 1, (N, 2))

        super(NNSensor, self).__init__(Xin=coords, k=k,
                                       rescale=False, center=False,
                                       plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        return {'k': self.k,
                'distributed': self.distributed,
                'seed': self.seed}
