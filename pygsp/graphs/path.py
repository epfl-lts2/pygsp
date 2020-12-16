# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Path(Graph):
    r"""Path graph.

    A signal on the path graph is akin to a 1-dimensional signal in classical
    signal processing.

    On the path graph, the graph Fourier transform (GFT) is the classical
    discrete cosine transform (DCT_).
    As the type-II DCT, the GFT assumes even boundary conditions on both sides.

    .. _DCT: https://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    N : int
        Number of vertices.

    See Also
    --------
    Ring : 1D line with periodic boundary conditions
    Grid2d : Kronecker product of two path graphs
    Comet : Generalization with a star at one end

    References
    ----------
    :cite:`strang1999dct` shows that each DCT basis contains the eigenvectors
    of a symmetric "second difference" matrix.
    They get the eight types of DCTs by varying the boundary conditions.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    >>> for i, directed in enumerate([False, True]):
    ...     G = graphs.Path(N=10, directed=directed)
    ...     _ = axes[i, 0].spy(G.W)
    ...     _ = G.plot(ax=axes[i, 1])

    The GFT of the path graph is the classical DCT.

    >>> from matplotlib import pyplot as plt
    >>> n_eigenvectors = 4
    >>> graph = graphs.Path(30)
    >>> fig, axes = plt.subplots(1, 2)
    >>> graph.set_coordinates('line1D')
    >>> graph.compute_fourier_basis()
    >>> _ = graph.plot(graph.U[:, :n_eigenvectors], ax=axes[0])
    >>> _ = axes[0].legend(range(n_eigenvectors))
    >>> _ = axes[1].plot(graph.e, '.')

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
