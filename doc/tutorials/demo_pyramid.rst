========
GSP Demo Pyramid
========

In this demonstration file, we show how to reduce a graph using the GSPBox. Then we apply the pyramid to simple signal.
To start open a python shell (IPython is recommended here) and import the required packages. You would probably also import numpy as you will need it to create matrices and arrays.

>>> import numpy as np
>>> from pygsp.graphs import Sensor, compute_fourier_basis, estimate_lmax
>>> from pygsp.reduction import kron_pyramid, pyramid_cell2coeff, pyramid_analysis, pyramid_synthesis

For this demo we will be using a Sensor graph with 512 nodes.

>>> G = graphs.Sensor(512, distribute=True)

The function kron_pyramid computes the graph pyramid for you:

>>> Gs = kron_pyramid(G, 5, epsilon=0.1)


>>> compute_fourier_basis(Gs)
>>> estimate_lmax(Gs)

>>> f = np.ones((G.N))
>>> f[np.arange(G.N/2)] = -1
>>> f = f + 10*Gs[0].U[:, 7]

>>> f2 = np.ones((G.N, 2))
>>> f2[np.arange(G.N/2)] = -1

>>> g = [lambda x: 5./(5 + x)]

>>> ca, pe = pyramid_analysis(Gs, f, filters=g, verbose=False)
>>> ca2, pe2 = pyramid_analysis(Gs, f2, filters=g, verbose=False)

>>> coeff = pyramid_cell2coeff(ca, pe)
>>> coeff2 = pyramid_cell2coeff(ca2, pe2)

>>> f_pred, _ = pyramid_synthesis(Gs, coeff, verbose=False)
>>> f_pred2, _ = pyramid_synthesis(Gs, coeff2, verbose=False)

>>> err = np.linalg.norm(f_pred-f)/np.linalg.norm(f)
>>> err2 = np.linalg.norm(f_pred2-f2)/np.linalg.norm(f2)
>>> print('erreur de f (1d) : {}'.format(err))
>>> print('erreur de f2 (2d) : {}'.format(err2))







You have now a graph structure ready to be used everywhere in the box! If you want to know more about the Graph class and it's subclasses you can check the online doc at : https://lts2.epfl.ch/pygsp/
You can also check the included methods for all graphs with the usual help function.

For the next steps of the demo, we will be using the logo graph bundled with the toolbox :

>>> G = pygsp.graphs.Logo()

You can now plot the graph:

>>> pygsp.plotting.plot(G, savefig=True, plot_name='doc/tutorials/img/logo')

.. image:: logo.*

Looks good isn't it? Now we can start to analyse the graph. The next step to compute Graph Fourier Transform or exact graph filtering is to precompute the Fourier basis of the graph. This operation can be very long as it needs to to fully diagonalize the Laplacian. Happily it is not needed to filter signal on graphs.

>>> pygsp.operators.compute_fourier_basis(G)

You can now access the eigenvalues of the fourier basis with G.e and the eigenvectors G.U, they look like sinuses on the graph.
Let's plot the second and third eigenvector, as the one is only constant.

>>> pygsp.plotting.plot_signal(G, G.U[:, 2], savefig=True, vertex_size=50, plot_name='doc/tutorials/img/logo_second_eigenvector')
>>> pygsp.plotting.plot_signal(G, G.U[:, 3], savefig=True, vertex_size=50, plot_name='doc/tutorials/img/logo_third_eigenvector')

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

>>> pygsp.plotting.plot(F,plot_eigenvalues=True, savefig=True, plot_name='doc/tutorials/img/low_pass_filter')

.. image:: img/low_pass_filter.*

Here's our low pass filter.


To accompain our new filter, let's create a nice signal on the logo by setting each letter to a certain value and then adding some random noise.

>>> f = np.zeros((G.N,))
>>> f[G.info['idx_g']-1] = - 1
>>> f[G.info['idx_s']-1] = 1
>>> f[G.info['idx_p']-1] = -0.5
>>> f += np.random.rand(G.N,)

The filter is plotted all along the spectrum of the graph, the cross at the bottom are the laplacian's eigenvalues. Those are the point where the continuous filter will be evaluated to create a discrete filter.
To apply it to a given signal, you only need to run:

>>> f2 = F.analysis(G, f)

Finally here's the noisy signal and the denoised version right under.

>>> pygsp.plotting.plot_signal(G, f, savefig=True, vertex_size=50, plot_name='doc/tutorials/img/noisy_logo')
>>> pygsp.plotting.plot_signal(G, f2, savefig=True, vertex_size=50, plot_name='doc/tutorials/img/denoised_logo')

.. image:: img/noisy_logo.*
.. image:: img/denoised_logo.*

So here are the basics for the PyGSP toolbox, if you want more informations you can check the doc at : #TODO.
Enjoy the toolbox!
