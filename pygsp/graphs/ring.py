# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from . import Graph  # prevent circular import in Python < 3.5


class Ring(Graph):
    r"""K-regular ring graph.

    A signal on the ring graph is akin to a 1-dimensional periodic signal in
    classical signal processing.

    On the ring graph, the graph Fourier transform (GFT) is the classical
    discrete Fourier transform (DFT_).
    Actually, the Laplacian of the ring graph is a `circulant matrix`_, and any
    circulant matrix is diagonalized by the DFT.

    .. _DFT: https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    .. _circulant matrix: https://en.wikipedia.org/wiki/Circulant_matrix

    Parameters
    ----------
    N : int
        Number of vertices.
    k : int
        Number of neighbors in each direction.

    See Also
    --------
    Path : 1D line with even boundary conditions
    Torus : Kronecker product of two ring graphs

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=10)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> _ = G.plot(ax=axes[1])

    The GFT of the ring graph is the classical DFT.

    >>> from matplotlib import pyplot as plt
    >>> n_eigenvectors = 4
    >>> graph = graphs.Ring(30)
    >>> fig, axes = plt.subplots(1, 2)
    >>> graph.set_coordinates('line1D')
    >>> graph.compute_fourier_basis()
    >>> _ = graph.plot(graph.U[:, :n_eigenvectors], ax=axes[0])
    >>> _ = axes[0].legend(range(n_eigenvectors))
    >>> _ = axes[1].plot(graph.e, '.')

    """

    def __init__(self, N=64, k=1, **kwargs):

        self.k = k

        if N < 3:
            # Asymmetric graph needed for 2 as 2 distances connect them.
            raise ValueError('There should be at least 3 vertices.')

        if 2*k > N:
            raise ValueError('Too many neighbors requested.')

        if 2*k == N:
            num_edges = N * (k - 1) + k
        else:
            num_edges = N * k

        i_inds = np.zeros((2 * num_edges))
        j_inds = np.zeros((2 * num_edges))

        tmpN = np.arange(N, dtype=int)
        for i in range(min(k, (N - 1) // 2)):
            i_inds[2*i * N + tmpN] = tmpN
            j_inds[2*i * N + tmpN] = np.remainder(tmpN + i + 1, N)
            i_inds[(2*i + 1)*N + tmpN] = np.remainder(tmpN + i + 1, N)
            j_inds[(2*i + 1)*N + tmpN] = tmpN

        if 2*k == N:
            i_inds[2*N*(k - 1) + tmpN] = tmpN
            i_inds[2*N*(k - 1) + tmpN] = np.remainder(tmpN + k + 1, N)

        W = sparse.csc_matrix((np.ones((2*num_edges)), (i_inds, j_inds)),
                              shape=(N, N))

        plotting = {'limits': np.array([-1, 1, -1, 1])}

        super(Ring, self).__init__(W, plotting=plotting, **kwargs)

        self.set_coordinates('ring2D')

    def _get_extra_repr(self):
        return dict(k=self.k)
