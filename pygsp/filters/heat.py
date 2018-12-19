# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Heat(Filter):
    r"""Design a filter bank of heat kernels.

    The (low-pass) heat kernel is defined in the spectral domain as

    .. math:: g_\tau(\lambda) = \exp(-\tau \lambda),

    where :math:`\lambda \in [0, 1]` are the normalized eigenvalues of the
    graph Laplacian, and :math:`\tau` is a parameter that captures both time
    and thermal diffusivity.

    The heat kernel is the fundamental solution to the heat equation

    .. math:: - \tau L f(t) = \partial_t f(t),

    where :math:`f: \mathbb{R}_+ \rightarrow \mathbb{R}^N` is the heat
    distribution over the graph at time :math:`t`. Given the initial condition
    :math:`f(0)`, the solution of the heat equation is expressed as

    .. math:: f(t) = e^{-\tau t L} f(0)
                   = U e^{-\tau t \Lambda} U^\top f(0)
                   = g_{\tau t}(L) f(0).

    The above is, by definition, the convolution of the signal :math:`f(0)`
    with the kernel :math:`g_{\tau t}(\lambda) = \exp(-\tau t \lambda)`.
    Hence, applying this filter to a signal simulates heat diffusion.

    Since the kernel is applied to the graph eigenvalues :math:`\lambda`, which
    can be interpreted as squared frequencies, it can also be considered as a
    generalization of the Gaussian kernel on graphs.

    Parameters
    ----------
    G : graph
    scale : float or iterable
        Scaling parameter. When solving heat diffusion, it encompasses both
        time and thermal diffusivity.
        If iterable, creates a filter bank with one filter per value.
    normalize : bool
        Whether to normalize the kernel to have unit L2 norm.
        The normalization needs the eigenvalues of the graph Laplacian.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Heat(G, scale=[5, 10, 100])
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    Heat diffusion from two sources on a grid.

    >>> import matplotlib.pyplot as plt
    >>> n_side = 11
    >>> graph = graphs.Grid2d(n_side)
    >>> graph.estimate_lmax()
    >>> sources = [
    ...     (n_side//4 * n_side) + (n_side//4),
    ...     (n_side*3//4 * n_side) + (n_side*3//4),
    ... ]
    >>> delta = np.zeros(graph.n_vertices)
    >>> delta[sources] = 5
    >>> steps = np.array([1, 5])
    >>> diffusivity = 10
    >>> g = filters.Heat(graph, scale=diffusivity*steps)
    >>> diffused = g.filter(delta)
    >>> fig, axes = plt.subplots(1, len(steps), figsize=(10, 4))
    >>> _ = fig.suptitle('Heat diffusion', fontsize=16)
    >>> for i, ax in enumerate(axes):
    ...     _ = graph.plot(diffused[:, i], highlight=sources,
    ...                    title='step {}'.format(steps[i]), ax=ax)
    ...     ax.set_aspect('equal', 'box')
    ...     ax.set_axis_off()

    Normalized heat kernel.

    >>> G = graphs.Logo()
    >>> G.compute_fourier_basis()
    >>> g = filters.Heat(G, scale=5)
    >>> y = g.evaluate(G.e)
    >>> print('norm: {:.2f}'.format(np.linalg.norm(y[0])))
    norm: 9.76
    >>> g = filters.Heat(G, scale=5, normalize=True)
    >>> y = g.evaluate(G.e)
    >>> print('norm: {:.2f}'.format(np.linalg.norm(y[0])))
    norm: 1.00

    """

    def __init__(self, G, scale=10, normalize=False):

        try:
            iter(scale)
        except TypeError:
            scale = [scale]

        self.scale = scale
        self.normalize = normalize

        def kernel(x, scale):
            return np.minimum(np.exp(-scale * x / G.lmax), 1)

        kernels = []
        for s in scale:
            norm = np.linalg.norm(kernel(G.e, s)) if normalize else 1
            kernels.append(lambda x, s=s, norm=norm: kernel(x, s) / norm)

        super(Heat, self).__init__(G, kernels)

    def _get_extra_repr(self):
        scale = '[' + ', '.join('{:.2f}'.format(s) for s in self.scale) + ']'
        return dict(scale=scale, normalize=self.normalize)
