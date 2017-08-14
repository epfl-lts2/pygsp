=======================================
Introduction to spectral graph wavelets
=======================================

Description
-----------

The wavelets are a special type of filterbank, in this demo we will show you how you can very easily construct a wavelet frame and apply it to a signal.

In this demo we will show you how to compute the wavelet coefficients of a graph and visualize them.
First let's import the toolbox, numpy and load a graph.

>>> import pygsp
>>> import numpy as np
>>> G = pygsp.graphs.Bunny()

This graph is a nearest-neighbor graph of a pointcloud of the Stanford bunny. It will allow us to get interesting visual results using wavelets.

At this stage we could compute the full Fourier basis using

>>> G.compute_fourier_basis()

but this would take a lot of time, and can be avoided by using Chebychev polynomials approximations.

Simple filtering
----------------

Before tackling wavelets, we can see the effect of one filter localized on the graph. So we can first design a few heat kernel filters

>>> taus = [1, 10, 25, 50]
>>> Hk = pygsp.filters.Heat(G, taus, normalize=False)

Let's now create a signal as a Kronecker located on one vertex (e.g. the vertex 83)

>>> S = np.zeros(G.N)
>>> vertex_delta = 83
>>> S[vertex_delta] = 1
>>> Sf_vec = Hk.analysis(S)
>>> Sf = Sf_vec.reshape((Sf_vec.size//len(taus), len(taus)), order='F')

Let's plot the signal:

>>> pygsp.plotting.plt_plot_signal(G, Sf[:,0], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/heat_tau_1')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,1], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/heat_tau_10')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,2], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/heat_tau_25')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,3], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/heat_tau_50')

.. figure:: img/heat_tau_1.*
    :alt: Tau = 1
    :align: center

    Heat tau = 1

.. figure:: img/heat_tau_10.*
    :alt: Tau = 10
    :align: center

    Heat tau = 10

.. figure:: img/heat_tau_25.*
    :alt: Tau = 25
    :align: center

    Heat tau = 25

.. figure:: img/heat_tau_50.*
    :alt: Tau = 50
    :align: center

    Heat tau = 50

Visualizing wavelets atoms
--------------------------

Let's now replace the Heat filter by a filter bank of wavelets. We can create a filter bank using one of the predefined filters such as pygsp.filters.MexicanHat.

>>> Nf = 6
>>> Wk = pygsp.filters.MexicanHat(G, Nf)

We can now plot the filter bank spectrum :

>>> Wk.plot(savefig=True, plot_name='doc/tutorials/img/mexican_hat')

.. figure:: img/mexican_hat.*
    :alt: Mexican Hat Wavelet filter
    :align: center

    Mexican Hat Wavelet filter

As we can see, the wavelets atoms are stacked on the low frequency part of the spectrum.
If we want to get a better coverage of the graph spectrum, we could have used the WarpedTranslates filter bank.

>>> S_vec = Wk.analysis(S)
>>> S = S_vec.reshape((S_vec.size//Nf, Nf), order='F')
>>> pygsp.plotting.plt_plot_signal(G, S[:, 0], savefig=True, plot_name='doc/tutorials/img/wavelet_filtering')


We can visualize the filtering by one atom the same way the did for the Heat kernel, by placing a Kronecker delta at one specific vertex.

>>> S = np.zeros((G.N * Nf, Nf))
>>> S[vertex_delta] = 1
>>> for i in range(Nf):
...     S[vertex_delta + i * G.N, i] = 1
>>> Sf = Wk.synthesis(S)

>>> pygsp.plotting.plt_plot_signal(G, Sf[:,0], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/wavelet_1')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,1], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/wavelet_2')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,2], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/wavelet_3')
>>> pygsp.plotting.plt_plot_signal(G, Sf[:,3], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/wavelet_4')

.. figure:: img/wavelet_1.*
.. figure:: img/wavelet_2.*
.. figure:: img/wavelet_3.*
.. figure:: img/wavelet_4.*

>>> G = pygsp.graphs.Bunny()
>>> Wk = pygsp.filters.MexicanHat(G, Nf)
>>> s_map = G.coords

>>> s_map_out = Wk.analysis(s_map)
>>> s_map_out = np.reshape(s_map_out, (G.N, Nf, 3))

>>> d = s_map_out[:, :, 0]**2 + s_map_out[:, :, 1]**2 + s_map_out[:, :, 2]**2
>>> d = np.sqrt(d)

>>> pygsp.plotting.plt_plot_signal(G, d[:, 1], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/curv_scale_1')
>>> pygsp.plotting.plt_plot_signal(G, d[:, 2], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/curv_scale_2')
>>> pygsp.plotting.plt_plot_signal(G, d[:, 3], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/curv_scale_3')
>>> pygsp.plotting.plt_plot_signal(G, d[:, 4], vertex_size=20, savefig=True, plot_name='doc/tutorials/img/curv_scale_4')

.. figure:: img/curv_scale_1.*
.. figure:: img/curv_scale_2.*
.. figure:: img/curv_scale_3.*
.. figure:: img/curv_scale_4.*
