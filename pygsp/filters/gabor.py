from pygsp import utils

from .filter import Filter  # prevent circular import in Python < 3.5


class Gabor(Filter):
    r"""Design a filter bank with a kernel centered at each frequency.

    Design a filter bank from translated versions of a mother filter.
    The mother filter is translated to each eigenvalue of the Laplacian.
    That is equivalent to convolutions with deltas placed at those eigenvalues.

    In classical image processing, a Gabor filter is a sinusoidal wave
    multiplied by a Gaussian function (here, the kernel). It analyzes whether
    there are any specific frequency content in the image in specific
    directions in a localized region around the point of analysis. This
    implementation for graph signals allows arbitrary (but isotropic) kernels.

    This filter bank can be used to compute the frequency content of a signal
    at each vertex. After filtering, one obtains a vertex-frequency
    representation :math:`Sf(i,k)` of a signal :math:`f` as

    .. math:: Sf(i, k) = \langle g_{i,k}, f \rangle,

    where :math:`g_{i,k}` is the mother kernel centered on eigenvalue
    :math:`\lambda_k` and localized on vertex :math:`v_i`.

    While :math:`g_{i,k}` should ideally be localized in both the spectral and
    vertex domains, that is impossible for some graphs due to the localization
    of some eigenvectors. See :attr:`pygsp.graphs.Graph.coherence`.

    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
    kernel : :class:`pygsp.filters.Filter`
        Kernel function to be centered at each graph frequency (eigenvalue of
        the graph Laplacian).

    See Also
    --------
    Modulation : Another way to translate a filter in the spectral domain.

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
    >>>
    >>> g1 = filters.Expwin(G, band_min=None, band_max=0, slope=3)
    >>> g2 = filters.Rectangular(G, band_min=-0.05, band_max=0.05)
    >>> g3 = filters.Heat(G, scale=10)
    >>>
    >>> fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    >>> for g, ax in zip([g1, g2, g3], axes):
    ...     g = filters.Gabor(G, g)
    ...     s = g.localize(G.N // 2, method='exact')
    ...     _ = g.plot(ax=ax[0], sum=False)
    ...     _ = G.plot(s, ax=ax[1])
    >>> fig.tight_layout()

    """

    def __init__(self, graph, kernel):
        if kernel.n_filters != 1:
            raise ValueError(
                "A kernel must be one filter. The passed "
                "filter bank {} has {}.".format(kernel, kernel.n_filters)
            )
        if kernel.G is not graph:
            raise ValueError(
                "The graph passed to this filter bank must "
                "be the one used to build the mother kernel."
            )

        kernels = []
        for i in range(graph.n_vertices):
            kernels.append(lambda x, i=i: kernel.evaluate(x - graph.e[i]))

        super().__init__(graph, kernels)

    def filter(self, s, method="exact", order=None):
        """TODO: indirection will be removed when poly filtering is merged."""
        return super().filter(s, method="exact")
