================
GSP Wavelet Demo
================

* Introduction to spectral graph wavelet whith the PyGSP

Description
-----------

The wavelets are a special type of filterbank, in this demo we will show you how you can very easily construct a wavelet frame and apply it to a signal.

In this demo we will show you how to compute the wavelet coefficients of a graph and visualize them.
First let's import the toolbox, numpy and load a graph.

>>> import pygsp
>>> import numpy as np
>>> G = pygsp.graphs.Bunny()

This graph is a nearest-neighbor graph of a pointcloud of the Stanford bunny. It will allow us ot get interesting visual results using wavelets.

At this stage we could compute the full Fourier basis using 

>>> pygsp.operators.compute_fourier_basis(G)

but this would take a lot of time, and can be avoided by using Chebychev polynomials approximations.

Simple filtering
----------------

Before tackling wavelets, we can see the effect of one filter localized on the graph. So we can first design a few heat kernel filters

>>> taus = [1, 10, 25, 50]
>>> Hk = pygsp.filters.Heat(G, taus)

Let's now create a signal as a Kronecker located on one vertex (e.g. the vertex 83)

>>> S = np.zeros(G.N)
>>> vertex_delta = 83
>>> S[vertex_delta] = 1
>>> Sf_vec = Hk.analysis(G, S)
>>> Sf = Sf_vec.reshape(Sf_vec.size/len(taus), len(taus))

Let's plot the signal:

>>> pygsp.plotting.plot_signal(G, Sf[:,0], savefig=True, plot_name='doc/tutorials/img/heat_tau_1')
>>> pygsp.plotting.plot_signal(G, Sf[:,1], savefig=True, plot_name='doc/tutorials/img/heat_tau_10')
>>> pygsp.plotting.plot_signal(G, Sf[:,2], savefig=True, plot_name='doc/tutorials/img/heat_tau_25')
>>> pygsp.plotting.plot_signal(G, Sf[:,3], savefig=True, plot_name='doc/tutorials/img/heat_tau_50')

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

>>> pygsp.plotting.plot(Wk, savefig=True, plot_name='doc/tutorials/img/mexican_hat')

.. figure:: img/mexican_hat.*
    :alt: Mexican Hat Wavelet filter
    :align: center

    Mexican Hat Wavelet filter

As we can see, the wavelets atoms are stacked on the low frequency part of the spectrum.
If we want to get a better coverage of the graph spectrum, we could have used the WarpedTranslates filter bank.

We can visualize the filtering by one atom the same way the did for the Heat kernel, by placing a Kronecker delta at one specific vertex.

>>> S = np.zeros((G.N * Nf, Nf))
>>> S[vertex_delta] = 1
>>> for i in range(Nf):
...     S[vertex_delta + i * G.N, i] = 1
>>> Sf = Wk.synthesis(G, S)

>>> pygsp.plotting.plot_signal(G, Sf[:,0], savefig=True, plot_name='doc/tutorials/img/wavelet_1')
>>> pygsp.plotting.plot_signal(G, Sf[:,1], savefig=True, plot_name='doc/tutorials/img/wavelet_2')
>>> pygsp.plotting.plot_signal(G, Sf[:,2], savefig=True, plot_name='doc/tutorials/img/wavelet_3')
>>> pygsp.plotting.plot_signal(G, Sf[:,3], savefig=True, plot_name='doc/tutorials/img/wavelet_4')

.. figure:: img/wavelet_1.*
.. figure:: img/wavelet_2.*
.. figure:: img/wavelet_3.*
.. figure:: img/wavelet_4.*
