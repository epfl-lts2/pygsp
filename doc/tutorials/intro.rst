=========================
Introduction to the PyGSP
=========================

This tutorial shows basic operations of the toolbox.
To start open a python shell (IPython is recommended here) and import the pygsp. You would probably also import numpy as you will need it to create matrices and arrays.

>>> import pygsp
>>> import numpy as np

The first step is to create a graph, there's a general class that can be used to generate graph from it's weight matrix.

>>> np.random.seed(42) # We will use a seed to make reproducible results
>>> W = np.random.rand(400, 400)
>>> G = pygsp.graphs.Graph(W)

You have now a graph structure ready to be used everywhere in the box! Check the :mod:`pygsp.graphs` module to know more about the Graph class and it's subclasses.
You can also check the included methods for all graphs with the usual help function.

For the next steps of the demo, we will be using the logo graph bundled with the toolbox :

>>> G = pygsp.graphs.Logo()

You can now plot the graph:

>>> G.plot(default_qtg=False, savefig=True, plot_name='doc/tutorials/img/logo')

.. image:: img/logo.*

Looks good isn't it? Now we can start to analyse the graph. The next step to compute Graph Fourier Transform or exact graph filtering is to precompute the Fourier basis of the graph. This operation can be very long as it needs to to fully diagonalize the Laplacian. Happily it is not needed to filter signal on graphs.

>>> G.compute_fourier_basis()

You can now access the eigenvalues of the fourier basis with G.e and the eigenvectors G.U, they look like sinuses on the graph.
Let's plot the second and third eigenvectors, as the first is constant.

>>> G.plot_signal(G.U[:, 1], vertex_size=50, default_qtg=False, savefig=True, plot_name='doc/tutorials/img/logo_second_eigenvector')
>>> G.plot_signal(G.U[:, 2], vertex_size=50, default_qtg=False, savefig=True, plot_name='doc/tutorials/img/logo_third_eigenvector')

.. figure:: img/logo_second_eigenvector.*

    Second eigenvector

.. figure:: img/logo_third_eigenvector.*

    Third eigenvector

Let's discover basic filters operations, filters are usually defined in the spectral domain.

First let's define a filter object:

>>> F = pygsp.filters.Filter(G)

And we can assign this function

.. math:: \begin{equation*} g(x) =\frac{1}{1+\tau x} \end{equation*}

to it:

>>> tau = 1
>>> g = lambda x: 1./(1. + tau * x)
>>> F.g = [g]

You can also put multiple functions in a list to define a filterbank!

>>> F.plot(plot_eigenvalues=True, savefig=True, plot_name='doc/tutorials/img/low_pass_filter')

.. image:: img/low_pass_filter.*

Here's our low pass filter.

To go with our new filter, let's create a nice signal on the logo by setting each letter to a certain value and then adding some random noise.

>>> f = np.zeros((G.N,))
>>> f[G.info['idx_g']-1] = - 1
>>> f[G.info['idx_s']-1] = 1
>>> f[G.info['idx_p']-1] = -0.5
>>> f += np.random.rand(G.N,)

The filter is plotted all along the spectrum of the graph, the cross at the bottom are the laplacian's eigenvalues. Those are the point where the continuous filter will be evaluated to create a discrete filter.
To apply it to a given signal, you only need to run:

>>> f2 = F.analysis(f)

Finally here's the noisy signal and the denoised version right under.

>>> G.plot_signal(f, vertex_size=50, default_qtg=False, savefig=True, plot_name='doc/tutorials/img/noisy_logo')
>>> G.plot_signal(f2, vertex_size=50, default_qtg=False, savefig=True, plot_name='doc/tutorials/img/denoised_logo')

.. image:: img/noisy_logo.*
.. image:: img/denoised_logo.*

So here are the basics for the PyGSP toolbox, please check the other tutorials or the reference guide for more.

Enjoy the toolbox!
