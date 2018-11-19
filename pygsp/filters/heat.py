# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Heat(Filter):
    r"""Design a filter bank of heat kernels.

    The (low-pass) heat kernel filter is defined in the spectral domain as

    .. math:: \hat{g}_\tau(\lambda) =
        \exp \left( -\tau \frac{\lambda}{\lambda_\text{max}} \right).

    The heat kernel is the fundamental solution to the heat equation

    .. math:: \tau L f(t) = - \partial_t f(t),

    where :math:`f: \mathbb{R}_+ \rightarrow \mathbb{R}^N`. Given the initial
    condition :math:`f(0)`, the solution of the heat equation is expressed as

    .. math:: f(t) = e^{-L \tau t} f(0)
                   = U e^{-\Lambda \tau t} U^\top f(0)
                   = K_t(L) f(0).

    The above is, by definition, the convolution of the signal :math:`f(0)`
    with the kernel :math:`K_t(\lambda) = \exp(-\tau t \lambda) = \hat{g}_\tau
    (t \lambda \lambda_\text{max})`.
    Hence, applying this filter to a signal simulates heat diffusion.

    Since the kernel is applied to the graph eigenvalues :math:`\Lambda`, which
    can be interpreted as squared frequencies, it can also be considered as a
    generalization of the Gaussian kernel on graphs.

    Parameters
    ----------
    G : graph
    tau : int or list of ints
        Scaling parameter. If a list, creates a filter bank with one filter per
        value of tau.
    normalize : bool
        Normalizes the kernel. Needs the eigenvalues.

    Examples
    --------

    Regular heat kernel.

    >>> G = graphs.Logo()
    >>> g = filters.Heat(G, tau=[5, 10])
    >>> print('{} filters'.format(g.Nf))
    2 filters
    >>> y = g.evaluate(G.e)
    >>> print('{:.2f}'.format(np.linalg.norm(y[0])))
    9.76

    Normalized heat kernel.

    >>> g = filters.Heat(G, tau=[5, 10], normalize=True)
    >>> y = g.evaluate(G.e)
    >>> print('{:.2f}'.format(np.linalg.norm(y[0])))
    1.00

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Heat(G, tau=[5, 10, 100])
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
    >>> delta = np.zeros(graph.n_nodes)
    >>> delta[sources] = 5
    >>> steps = np.array([1, 5])
    >>> g = filters.Heat(graph, tau=10*steps)
    >>> diffused = g.filter(delta)
    >>> fig, axes = plt.subplots(1, len(steps), figsize=(10, 4))
    >>> _ = fig.suptitle('Heat diffusion', fontsize=16)
    >>> for i, ax in enumerate(axes):
    ...     _ = graph.plot(diffused[:, i], highlight=sources,
    ...                    title='step {}'.format(steps[i]), ax=ax)
    ...     ax.set_aspect('equal', 'box')
    ...     ax.set_axis_off()

    """

    def __init__(self, G, tau=10, normalize=False):

        try:
            iter(tau)
        except TypeError:
            tau = [tau]

        self.tau = tau
        self.normalize = normalize

        def kernel(x, t):
            return np.minimum(np.exp(-t * x / G.lmax), 1)

        kernels = []
        for t in tau:
            norm = np.linalg.norm(kernel(G.e, t)) if normalize else 1
            kernels.append(lambda x, t=t, norm=norm: kernel(x, t) / norm)

        super(Heat, self).__init__(G, kernels)

    def _get_extra_repr(self):
        tau = '[' + ', '.join('{:.2f}'.format(t) for t in self.tau) + ']'
        return dict(tau=tau, normalize=self.normalize)
