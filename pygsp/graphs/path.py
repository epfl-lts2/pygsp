# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Path(Graph):
    r"""Path graph.

    Parameters
    ----------
    N : int
        Number of vertices.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    >>> for i, directed in enumerate([False, True]):
    ...     G = graphs.Path(N=10, directed=directed)
    ...     _ = axes[i, 0].spy(G.W)
    ...     _ = G.plot(ax=axes[i, 1])

    References
    ----------
    See :cite:`strang1999discrete` for more informations.

    """

    def __init__(self, N=16, directed=False, **kwargs):

        self.directed = directed
        if directed:
            sources = np.arange(0, N-1)
            targets = np.arange(1, N)
            n_edges = N - 1
        else:
            sources = np.concatenate((np.arange(0, N-1), np.arange(1, N)))
            targets = np.concatenate((np.arange(1, N), np.arange(0, N-1)))
            n_edges = 2 * (N - 1)
        weights = np.ones(n_edges)
        W = sparse.csr_matrix((weights, (sources, targets)), shape=(N, N))
        plotting = {"limits": np.array([-1, N, -1, 1])}

        super(Path, self).__init__(W, plotting=plotting, **kwargs)

        self.set_coordinates('line2D')

    def _get_extra_repr(self):
        return dict(directed=self.directed)
