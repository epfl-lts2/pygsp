# -*- coding: utf-8 -*-

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class Gabor(Filter):
    r"""Design a filter bank where a kernel is centered at each frequency.

    In classical image processing, a Gabor filter is a sinusoidal wave
    multiplied by a Gaussian function (the kernel). It analyzes whether there
    are any specific frequency content in the image in specific directions in a
    localized region around the point of analysis. This implementation for
    graph signals allows arbitrary (but isotropic) kernels.

    This filter bank is used to compute the frequency content at each vertex, a
    kind of vertex-frequency analysis, or windowed graph Fourier transform.
    See :meth:`pygsp.graphs.Graph.gft_windowed_gabor`.

    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
    kernel : callable, can be a :class:`pygsp.filters.Filter`
        Kernel function to be centered at each graph frequency (eigenvalue of
        the graph Laplacian).

    Notes
    -----
    The eigenvalues of the graph Laplacian (i.e., the Fourier basis) are needed
    to center the kernels.

    Examples
    --------

    Filter bank's representation in Fourier and time (path graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Path(N=7)
    >>> G.compute_fourier_basis()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Expwin(G, band_min=0, band_max=0, slope=10)
    >>> g = filters.Gabor(G, g)
    >>> s = g.localize(G.N // 2, method='exact')
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0], sum=False)
    >>> _ = G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, graph, kernel):

        kernels = []
        for i in range(graph.n_nodes):
            kernels.append(lambda x, i=i: kernel(x - graph.e[i]))

        super(Gabor, self).__init__(graph, kernels)
