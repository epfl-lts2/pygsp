# -*- coding: utf-8 -*-

from __future__ import division

from functools import partial

import numpy as np

from . import Filter  # prevent circular import in Python < 3.5


class Wave(Filter):
    r"""Design a filter bank of wave kernels.

    The wave kernel is defined in the spectral domain as

    .. math:: g_{\tau, t}(\lambda) = \cos \left( t
        \arccos \left( 1 - \frac{\tau^2}{2} \lambda \right) \right),

    where :math:`\lambda \in [0, 1]` are the normalized eigenvalues of the
    graph Laplacian, :math:`t` is time, and :math:`\tau` is the propagation
    speed.

    The wave kernel is the fundamental solution to the wave equation

    .. math:: - \tau^2 L f(t) = \partial_{tt} f(t),

    where :math:`f: \mathbb{R}_+ \rightarrow \mathbb{R}^N` models, for example,
    the mechanical displacement of a wave on a graph. Given the initial
    condition :math:`f(0)` and assuming a vanishing initial velocity, i.e., the
    first derivative in time of the initial distribution equals zero, the
    solution of the wave equation is expressed as

    .. math:: f(t) = U g_{\tau, t}(\Lambda) U^\top f(0)
                   = g_{\tau, t}(L) f(0).

    The above is, by definition, the convolution of the signal :math:`f(0)`
    with the kernel :math:`g_{\tau, t}`.
    Hence, applying this filter to a signal simulates wave propagation.

    Parameters
    ----------
    G : graph
    time : float or iterable
        Time step.
        If iterable, creates a filter bank with one filter per value.
    speed : float or iterable
        Propagation speed, bounded by 0 (included) and 2 (excluded).
        If iterable, creates a filter bank with one filter per value.

    References
    ----------
    :cite:`grassi2016timevertex`, :cite:`grassi2018timevertex`

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Wave(G, time=[5, 15], speed=1)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    Wave propagation from two sources on a grid.

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
    >>> steps = np.array([5, 10])
    >>> g = filters.Wave(graph, time=steps, speed=1)
    >>> propagated = g.filter(delta)
    >>> fig, axes = plt.subplots(1, len(steps), figsize=(10, 4))
    >>> _ = fig.suptitle('Wave propagation', fontsize=16)
    >>> for i, ax in enumerate(axes):
    ...     _ = graph.plot(propagated[:, i], highlight=sources,
    ...                    title='step {}'.format(steps[i]), ax=ax)
    ...     ax.set_aspect('equal', 'box')
    ...     ax.set_axis_off()

    """

    def __init__(self, G, time=10, speed=1):

        try:
            iter(time)
        except TypeError:
            time = [time]
        try:
            iter(speed)
        except TypeError:
            speed = [speed]

        self.time = time
        self.speed = speed

        if len(time) != len(speed):
            if len(speed) == 1:
                speed = speed * len(time)
            elif len(time) == 1:
                time = time * len(speed)
            else:
                raise ValueError('If both parameters are iterable, '
                                 'they should have the same length.')

        if np.any(np.asanyarray(speed) >= 2):
            raise ValueError('The wave propagation speed should be in [0, 2[')

        def kernel(x, time, speed):
            return np.cos(time * np.arccos(1 - speed**2 * x / G.lmax / 2))

        kernels = [partial(kernel, time=t, speed=s)
                   for t, s in zip(time, speed)]

        super(Wave, self).__init__(G, kernels)

    def _get_extra_repr(self):
        time = '[' + ', '.join('{:.2f}'.format(t) for t in self.time) + ']'
        speed = '[' + ', '.join('{:.2f}'.format(s) for s in self.speed) + ']'
        return dict(time=time, speed=speed)
