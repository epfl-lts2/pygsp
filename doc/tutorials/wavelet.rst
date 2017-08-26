=======================================
Introduction to spectral graph wavelets
=======================================

This tutorial will show you how to easily construct a wavelet_ frame, a kind of
filter bank, and apply it to a signal. This tutorial will walk you into
computing the wavelet coefficients of a graph, visualizing filters in the
vertex domain, and using the wavelets to estimate the curvature of a 3D shape.

.. _wavelet: https://en.wikipedia.org/wiki/Wavelet

As usual, we first have to import some packages.

.. plot::
    :context: reset

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pygsp import graphs, filters, plotting, utils
    >>> plotting.BACKEND = 'matplotlib'

Then we can load a graph. The graph we'll use is a nearest-neighbor graph of a
point cloud of the Stanford bunny. It will allow us to get interesting visual
results using wavelets.

.. plot::
    :context: close-figs

    >>> G = graphs.Bunny()

.. note::
    At this stage we could compute the Fourier basis using
    :meth:`pygsp.graphs.Graph.compute_fourier_basis`, but this would take some
    time, and can be avoided with a Chebychev polynomials approximation to
    graph filtering. See the documentation of the
    :meth:`pygsp.filters.Filter.analysis` filtering function and
    :cite:`hammond2011wavelets` for details on how it is down.

Simple filtering: heat diffusion
--------------------------------

Before tackling wavelets, let's observe the effect of a single filter on a
graph signal. We first design a few heat kernel filters, each with a different
scale.

.. plot::
    :context: close-figs

    >>> taus = [10, 25, 50]
    >>> g = filters.Heat(G, taus)

Let's create a signal as a Kronecker delta located on one vertex, e.g. the
vertex 20. That signal is our heat source.

.. plot::
    :context: close-figs

    >>> s = np.zeros(G.N)
    >>> delta = 20
    >>> s[delta] = 1

We can now simulate heat diffusion by filtering our signal `s` with each of our
heat kernels.

.. plot::
    :context: close-figs

    >>> s = g.analysis(s, method='chebyshev')
    >>> s = utils.vec2mat(s, g.Nf)

And finally plot the filtered signal showing heat diffusion at different
scales.

.. plot::
    :context: close-figs

    >>> fig = plt.figure(figsize=(10, 3))
    >>> for i in range(g.Nf):
    ...     ax = fig.add_subplot(1, g.Nf, i+1, projection='3d')
    ...     G.plot_signal(s[:, i], vertex_size=20, colorbar=False, ax=ax)
    ...     title = r'Heat diffusion, $\tau={}$'.format(taus[i])
    ...     ax.set_title(title)  #doctest:+SKIP
    ...     ax.set_axis_off()
    >>> fig.tight_layout()  # doctest:+SKIP

.. note::
    The :meth:`pygsp.filters.Filter.localize` method can be used to visualize a
    filter in the vertex domain instead of doing it manually.

Visualizing wavelets atoms
--------------------------

Let's now replace the Heat filter by a filter bank of wavelets. We can create a
filter bank using one of the predefined filters, such as
:class:`pygsp.filters.MexicanHat` to design a set of `Mexican hat wavelets`_.

.. _Mexican hat wavelets:
    https://en.wikipedia.org/wiki/Mexican_hat_wavelet

.. plot::
    :context: close-figs

    >>> g = filters.MexicanHat(G, Nf=6)  # Nf = 6 filters in the filter bank.

Then plot the frequency response of those filters.

.. plot::
    :context: close-figs

    >>> fig, ax = plt.subplots(figsize=(10, 5))
    >>> g.plot(ax=ax)
    >>> ax.set_title('Filter bank of mexican hat wavelets')  # doctest:+SKIP

.. note::
    We can see that the wavelet atoms are stacked on the low frequency part of
    the spectrum. A better coverage could be obtained by adapting the filter
    bank with :class:`pygsp.filters.WarpedTranslates` or by using another
    filter bank like :class:`pygsp.filters.Itersine`.

We can visualize the filtering by one atom as we did with the heat kernel, by
filtering a Kronecker delta placed at one specific vertex.

.. plot::
    :context: close-figs

    >>> s = np.zeros((G.N * g.Nf, g.Nf))
    >>> s[delta] = 1
    >>> for i in range(g.Nf):
    ...     s[delta + i * G.N, i] = 1
    >>> s = g.synthesis(s)
    >>>
    >>> fig = plt.figure(figsize=(10, 7))
    >>> for i in range(4):
    ... 
    ...     # Clip the signal.
    ...     mu = np.mean(s[:, i])
    ...     sigma = np.std(s[:, i])
    ...     limits = [mu-4*sigma, mu+4*sigma]
    ... 
    ...     ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ...     G.plot_signal(s[:, i], vertex_size=20, limits=limits, ax=ax)
    ...     ax.set_title('Wavelet {}'.format(i+1))  # doctest:+SKIP
    ...     ax.set_axis_off()
    >>> fig.tight_layout()  # doctest:+SKIP

Curvature estimation
--------------------

As a last and more applied example, let us try to estimate the curvature of the
underlying 3D model by only using spectral filtering on the nearest-neighbor
graph formed by its point cloud.

A simple way to accomplish that is to use the coordinates map :math:`[x, y, z]`
and filter it using the above defined wavelets. Doing so gives us a
3-dimensional signal
:math:`[g_i(L)x, g_i(L)y, g_i(L)z], \ i \in [0, \ldots, N_f]`
which describes variation along the 3 coordinates.

.. plot::
    :context: close-figs

    >>> s = G.coords
    >>> s = g.analysis(s)
    >>> s = utils.vec2mat(s, g.Nf)

The curvature is then estimated by taking the :math:`\ell_1` or :math:`\ell_2`
norm of the filtered signal.

.. plot::
    :context: close-figs

    >>> s = np.linalg.norm(s, ord=2, axis=2)

Let's finally plot the result to observe that we indeed have a measure of the
curvature at different scales.

.. plot::
    :context: close-figs

    >>> fig = plt.figure(figsize=(10, 7))
    >>> for i in range(4):
    ...     ax = fig.add_subplot(2, 2, i+1, projection='3d')
    ...     G.plot_signal(s[:, i], vertex_size=20, ax=ax)
    ...     title = 'Curvature estimation (scale {})'.format(i+1)
    ...     ax.set_title(title)  # doctest:+SKIP
    ...     ax.set_axis_off()
    >>> fig.tight_layout()  # doctest:+SKIP
