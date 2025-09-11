import numpy as np
from scipy import interpolate

from .filter import Filter  # prevent circular import in Python < 3.5


class Modulation(Filter):
    r"""Design a filter bank with a kernel centered at each frequency.

    Design a filter bank from translated versions of a mother filter.
    The mother filter is translated to each eigenvalue of the Laplacian via
    modulation. A signal is modulated by multiplying it with an eigenvector.
    Similarly to localization, it is an element-wise multiplication of a kernel
    with the columns of :attr:`pygsp.graphs.Graph.U`, i.e., the eigenvectors,
    in the vertex domain.

    This filter bank can be used to compute the frequency content of a signal
    at each vertex. After filtering, one obtains a vertex-frequency
    representation :math:`Sf(i,k)` of a signal :math:`f` as

    .. math:: Sf(i, k) = \langle g_{i,k}, f \rangle,

    where :math:`g_{i,k}` is the mother kernel modulated in the spectral domain
    by the eigenvector :math:`u_k`, and localized on vertex :math:`v_i`.

    While :math:`g_{i,k}` should ideally be localized in both the spectral and
    vertex domains, that is impossible for some graphs due to the localization
    of some eigenvectors. See :attr:`pygsp.graphs.Graph.coherence`.

    As modulation and localization don't commute, one can define the frame as
    :math:`g_{i,k} = T_i M_k g` (modulation first) or :math:`g_{i,k} = M_k T_i
    g` (localization first). Localization first usually gives better results.
    When localizing first, the obtained vertex-frequency representation is a
    generalization to graphs of the windowed graph Fourier transform. Indeed,

    .. math:: Sf(i, k) = \langle f^\text{win}_i, u_k \rangle

    is the graph Fourier transform of the windowed signal :math:`f^\text{win}`.
    The signal :math:`f` is windowed in the vertex domain by a point-wise
    multiplication with the localized kernel :math:`T_i g`.

    When localizing first, the spectral representation of the filter bank is
    different for every localization. As such, we always evaluate the filter in
    the spectral domain with modulation first. Moreover, the filter bank is
    only defined at the eigenvalues (as modulation is done with discrete
    eigenvectors). Evaluating it elsewhere returns NaNs.

    Parameters
    ----------
    graph : :class:`pygsp.graphs.Graph`
    kernel : :class:`pygsp.filters.Filter`
        Kernel function to be modulated.
    modulation_first : bool
        First modulate then localize the kernel if True, first localize then
        modulate if False. The two operators do not commute. This setting only
        applies to :meth:`filter`. :meth:`evaluate` only performs modulation,
        as the filter would otherwise have a different spectrum depending on
        where it is localized.

    See Also
    --------
    Gabor : Another way to translate a filter in the spectral domain.

    Notes
    -----
    The eigenvalues of the graph Laplacian (i.e., the Fourier basis) are needed
    to modulate the kernels.

    References
    ----------
    See :cite:`shuman2016vertexfrequency` for details on this vertex-frequency
    representation of graph signals.

    Examples
    --------

    Vertex-frequency representations.
    Modulating first doesn't produce sufficiently localized filters.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Path(90)
    >>> G.compute_fourier_basis()
    >>>
    >>> # Design the filter banks.
    >>> g = filters.Heat(G, 500)
    >>> g1 = filters.Modulation(G, g, modulation_first=False)
    >>> g2 = filters.Modulation(G, g, modulation_first=True)
    >>> _ = g1.plot(sum=False, labels=False)
    >>>
    >>> # Signal.
    >>> s = np.empty(G.N)
    >>> s[:30] = G.U[:30, 10]
    >>> s[30:60] = G.U[30:60, 60]
    >>> s[60:] = G.U[60:, 30]
    >>> G.set_coordinates('line1D')
    >>> _ = G.plot(s)
    >>>
    >>> # Filter with both filter banks.
    >>> s1 = g1.filter(s)
    >>> s2 = g2.filter(s)
    >>>
    >>> # Visualize the vertex-frequency representation of the signal.
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].imshow(np.abs(s1.T)**2)
    >>> _ = axes[1].imshow(np.abs(s2.T)**2)
    >>> _ = axes[0].set_title('localization then modulation')
    >>> _ = axes[1].set_title('modulation then localization')
    >>> ticks = [0, G.N//2, G.N-1]
    >>> labels = ['{:.1f}'.format(e) for e in G.e[ticks]]
    >>> _ = axes[0].set_yticks(ticks)
    >>> _ = axes[1].set_yticks([])
    >>> _ = axes[0].set_yticklabels(labels)
    >>> _ = axes[0].set_ylabel('graph frequency')
    >>> _ = axes[0].set_xlabel('node')
    >>> _ = axes[1].set_xlabel('node')
    >>> _ = axes[0].set_xticks(ticks)
    >>> _ = axes[1].set_xticks(ticks)
    >>> fig.tight_layout()
    >>>
    >>> # Reconstruction.
    >>> s = g2.filter(s2)
    >>> _ = G.plot(s)

    """

    def __init__(self, graph, kernel, modulation_first=False):
        self.G = graph
        self._kernels = kernel
        self._modulation_first = modulation_first

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

        self.n_features_in, self.n_features_out = (1, graph.n_vertices)
        self.n_filters = self.n_features_in * self.n_features_out
        self.Nf = self.n_filters  # TODO: kept for backward compatibility only.

    def evaluate(self, x):
        """TODO: will become _evaluate once polynomial filtering is merged."""

        if not hasattr(self, "_coefficients"):
            # Graph Fourier transform -> modulation -> inverse GFT.
            c = self.G.igft(self._kernels.evaluate(self.G.e).squeeze())
            c = np.sqrt(self.G.n_vertices) * self.G.U * c[:, np.newaxis]
            self._coefficients = self.G.gft(c)

        shape = x.shape
        x = x.flatten()
        y = np.full((self.n_features_out, x.size), np.nan)
        for i in range(len(x)):
            query = self._coefficients[x[i] == self.G.e]
            if len(query) != 0:
                y[:, i] = query[0]
        return y.reshape((self.n_features_out,) + shape)

    def filter(self, s, method="exact", order=None):
        """TODO: indirection will be removed when poly filtering is merged.
        TODO: with _filter and shape handled in Filter.filter, synthesis will work.
        """
        if self._modulation_first:
            return super().filter(s, method="exact")
        else:
            # The dot product with each modulated kernel is equivalent to the
            # GFT, as for the localization and the IGFT.
            y = np.empty((self.G.n_vertices, self.G.n_vertices))
            for i in range(self.G.n_vertices):
                x = s * self._kernels.localize(i)
                y[i] = np.sqrt(self.G.n_vertices) * self.G.gft(x)
            return y
