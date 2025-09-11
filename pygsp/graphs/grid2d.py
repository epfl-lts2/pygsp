import numpy as np
from scipy import sparse

from pygsp import utils

from .graph import Graph  # prevent circular import in Python < 3.5


class Grid2d(Graph):
    r"""2-dimensional grid graph.

    On the 2D grid, the graph Fourier transform (GFT) is the Kronecker product
    between the GFT of two :class:`~pygsp.graphs.Path` graphs.

    Parameters
    ----------
    N1 : int
        Number of vertices along the first dimension.
    N2 : int
        Number of vertices along the second dimension. Default is ``N1``.
    diagonal : float
        Value of the diagnal edges. Default is ``0.0``

    See Also
    --------
    Path : 1D line with even boundary conditions
    Torus : Kronecker product of two ring graphs
    Grid2dImgPatches

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Grid2d(N1=5, N2=4)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, N1=16, N2=None, diagonal=0.0, **kwargs):
        if N2 is None:
            N2 = N1

        self.N1 = N1
        self.N2 = N2

        N = N1 * N2

        # Filling up the weight matrix this way is faster than
        # looping through all the grid points:
        diag_1 = np.ones(N - 1)
        diag_1[(N2 - 1) :: N2] = 0
        diag_2 = np.ones(N - N2)

        W = sparse.diags(
            diagonals=[diag_1, diag_2],
            offsets=[-1, -N2],
            shape=(N, N),
            format="csr",
            dtype="float",
        )

        if min(N1, N2) > 1 and diagonal != 0.0:
            # Connecting node with they diagonal neighbours
            diag_3 = np.full(N - N2 - 1, diagonal)
            diag_4 = np.full(N - N2 + 1, diagonal)
            diag_3[N2 - 1 :: N2] = 0
            diag_4[0::N2] = 0
            D = sparse.diags(
                diagonals=[diag_3, diag_4],
                offsets=[-N2 - 1, -N2 + 1],
                shape=(N, N),
                format="csr",
                dtype="float",
            )
            W += D

        W = utils.symmetrize(W, method="tril")

        x = np.kron(np.ones((N1, 1)), (np.arange(N2) / float(N2)).reshape(N2, 1))
        y = np.kron(np.ones((N2, 1)), np.arange(N1) / float(N1)).reshape(N, 1)
        y = np.sort(y, axis=0)[::-1]
        coords = np.concatenate((x, y), axis=1)

        plotting = {
            "limits": np.array([-1.0 / N2, 1 + 1.0 / N2, 1.0 / N1, 1 + 1.0 / N1])
        }

        super().__init__(W, coords=coords, plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        return dict(N1=self.N1, N2=self.N2)
