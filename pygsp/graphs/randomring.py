# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class RandomRing(Graph):
    r"""Ring graph with randomly sampled vertices.

    Parameters
    ----------
    N : int
        Number of vertices.
    angles : array_like, optional
        The angular coordinate, in :math:`[0, 2\pi]`, of the vertices.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.RandomRing(N=10, seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> _ = G.plot(ax=axes[1])
    >>> _ = axes[1].set_xlim(-1.1, 1.1)
    >>> _ = axes[1].set_ylim(-1.1, 1.1)

    """

    def __init__(self, N=64, angles=None, seed=None, **kwargs):

        self.seed = seed

        rs = np.random.RandomState(seed)
        angles = np.sort(rs.uniform(0, 2*np.pi, size=N), axis=0)
        self.angles = angles

        rows = range(0, N-1)
        cols = range(1, N)
        weights = np.diff(angles)

        # Close the loop.
        rows = np.concatenate((rows, [0]))
        cols = np.concatenate((cols, [N-1]))
        weights = np.concatenate((weights, [2*np.pi + angles[0] - angles[-1]]))

        W = sparse.coo_matrix((weights, (rows, cols)), shape=(N, N))
        W = utils.symmetrize(W, method='triu')

        # Width as the expected angle. All angles are equal to that value when
        # the ring is uniformly sampled.
        width = 2 * np.pi / N
        assert (W.data.mean() - width) < 1e-10
        # TODO: why this kernel ? It empirically produces eigenvectors closer
        # to the sines and cosines.
        W.data = width / W.data

        coords = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        plotting = {'limits': np.array([-1, 1, -1, 1])}

        # TODO: save angle and 2D position as graph signals
        super(RandomRing, self).__init__(W=W, coords=coords, plotting=plotting,
                                         **kwargs)

    def _get_extra_repr(self):
        return dict(seed=self.seed)
