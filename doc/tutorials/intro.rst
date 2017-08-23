=========================
Introduction to the PyGSP
=========================

This tutorial shows basic operations of the toolbox.
To start open a python shell (IPython is recommended here) and import the pygsp. You would probably also import numpy as you will need it to create matrices and arrays.

.. plot::
    :context: reset

    >>> import numpy as np
    >>> from pygsp import graphs, filters, plotting
    >>> plotting.BACKEND = 'matplotlib'

The first step is to create a graph, there's a general class that can be used to generate graph from it's weight matrix.

.. plot::
    :context: close-figs

    >>> np.random.seed(42) # We will use a seed to make reproducible results
    >>> W = np.random.rand(400, 400)
    >>> G = graphs.Graph(W)

You have now a graph structure ready to be used everywhere in the box! Check the :mod:`pygsp.graphs` module to know more about the Graph class and it's subclasses.
You can also check the included methods for all graphs with the usual help function.

For the next steps of the demo, we will be using the logo graph bundled with the toolbox :

.. plot::
    :context: close-figs

    >>> G = graphs.Logo()

You can now plot the graph:

.. plot::
    :context: close-figs

    >>> G.plot()

Looks good isn't it? Now we can start to analyse the graph. The next step to compute Graph Fourier Transform or exact graph filtering is to precompute the Fourier basis of the graph. This operation can be very long as it needs to to fully diagonalize the Laplacian. Happily it is not needed to filter signal on graphs.

.. plot::
    :context: close-figs

    >>> G.compute_fourier_basis()

You can now access the eigenvalues of the fourier basis with G.e and the eigenvectors G.U, they look like sinuses on the graph.
Let's plot the second and third eigenvectors, as the first is constant.

.. plot::
    :context: close-figs

    >>> G.plot_signal(G.U[:, 1], vertex_size=50)
    >>> G.plot_signal(G.U[:, 2], vertex_size=50)

Let's discover basic filters operations, filters are usually defined in the spectral domain.

Given the transfer function

.. math:: \begin{equation*} g(x) =\frac{1}{1+\tau x} \end{equation*},

let's define a filter object:

.. plot::
    :context: close-figs

    >>> tau = 1
    >>> def g(x):
    ...     return 1. / (1. + tau * x)
    >>> F = filters.Filter(G, g)

You can also put multiple functions in a list to define a filterbank!

.. plot::
    :context: close-figs

    >>> F.plot(plot_eigenvalues=True)

Here's our low pass filter.

To go with our new filter, let's create a nice signal on the logo by setting each letter to a certain value and then adding some random noise.

.. plot::
    :context: close-figs

    >>> f = np.zeros((G.N,))
    >>> f[G.info['idx_g']-1] = - 1
    >>> f[G.info['idx_s']-1] = 1
    >>> f[G.info['idx_p']-1] = -0.5
    >>> f += np.random.rand(G.N,)

The filter is plotted all along the spectrum of the graph, the cross at the bottom are the laplacian's eigenvalues. Those are the point where the continuous filter will be evaluated to create a discrete filter.
To apply it to a given signal, you only need to run:

.. plot::
    :context: close-figs

    >>> f2 = F.analysis(f)

Finally here's the noisy signal and the denoised version right under.

.. plot::
    :context: close-figs

    >>> G.plot_signal(f, vertex_size=50)
    >>> G.plot_signal(f2, vertex_size=50)

So here are the basics for the PyGSP toolbox, please check the other tutorials or the reference guide for more.

Enjoy the toolbox!
