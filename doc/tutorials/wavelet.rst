=======================================
Introduction to spectral graph wavelets
=======================================

Description
-----------

The wavelets are a special type of filterbank, in this demo we will show you how you can very easily construct a wavelet frame and apply it to a signal.

In this demo we will show you how to compute the wavelet coefficients of a graph and visualize them.
First let's import the toolbox, numpy and load a graph.

.. plot::
    :context: reset

    >>> import numpy as np
    >>> from pygsp import graphs, filters
    >>> G = graphs.Bunny()

This graph is a nearest-neighbor graph of a pointcloud of the Stanford bunny. It will allow us to get interesting visual results using wavelets.

At this stage we could compute the full Fourier basis using

.. plot::
    :context: close-figs

    >>> G.compute_fourier_basis()

but this would take a lot of time, and can be avoided by using Chebychev polynomials approximations.

Simple filtering
----------------

Before tackling wavelets, we can see the effect of one filter localized on the graph. So we can first design a few heat kernel filters

.. plot::
    :context: close-figs

    >>> taus = [1, 10, 25, 50]
    >>> Hk = filters.Heat(G, taus, normalize=False)

Let's now create a signal as a Kronecker located on one vertex (e.g. the vertex 83)

.. plot::
    :context: close-figs

    >>> S = np.zeros(G.N)
    >>> vertex_delta = 83
    >>> S[vertex_delta] = 1
    >>> Sf_vec = Hk.analysis(S)
    >>> Sf = Sf_vec.reshape((Sf_vec.size//len(taus), len(taus)), order='F')

Let's plot the signal:

.. plot::
    :context: close-figs

    >>> G.plot_signal(Sf[:,0], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,1], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,2], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,3], vertex_size=20, default_qtg=False)

Visualizing wavelets atoms
--------------------------

Let's now replace the Heat filter by a filter bank of wavelets. We can create a filter bank using one of the predefined filters such as :func:`pygsp.filters.MexicanHat`.

.. plot::
    :context: close-figs

    >>> Nf = 6
    >>> Wk = filters.MexicanHat(G, Nf)

We can now plot the filter bank spectrum :

.. plot::
    :context: close-figs

    >>> Wk.plot()

As we can see, the wavelets atoms are stacked on the low frequency part of the spectrum.
If we want to get a better coverage of the graph spectrum, we could have used the WarpedTranslates filter bank.

.. plot::
    :context: close-figs

    >>> S_vec = Wk.analysis(S)
    >>> S = S_vec.reshape((S_vec.size//Nf, Nf), order='F')
    >>> G.plot_signal(S[:, 0], default_qtg=False)

We can visualize the filtering by one atom the same way the did for the Heat kernel, by placing a Kronecker delta at one specific vertex.

.. plot::
    :context: close-figs

    >>> S = np.zeros((G.N * Nf, Nf))
    >>> S[vertex_delta] = 1
    >>> for i in range(Nf):
    ...     S[vertex_delta + i * G.N, i] = 1
    >>> Sf = Wk.synthesis(S)
    >>>
    >>> G.plot_signal(Sf[:,0], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,1], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,2], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(Sf[:,3], vertex_size=20, default_qtg=False)

.. plot::
    :context: close-figs

    >>> G = graphs.Bunny()
    >>> Wk = filters.MexicanHat(G, Nf)
    >>> s_map = G.coords
    >>>
    >>> s_map_out = Wk.analysis(s_map)
    >>> s_map_out = np.reshape(s_map_out, (G.N, Nf, 3))
    >>>
    >>> d = s_map_out[:, :, 0]**2 + s_map_out[:, :, 1]**2 + s_map_out[:, :, 2]**2
    >>> d = np.sqrt(d)
    >>>
    >>> G.plot_signal(d[:, 1], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(d[:, 2], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(d[:, 3], vertex_size=20, default_qtg=False)
    >>> G.plot_signal(d[:, 4], vertex_size=20, default_qtg=False)
