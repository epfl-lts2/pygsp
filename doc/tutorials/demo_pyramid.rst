========
GSP Demo Pyramid
========

In this demonstration file, we show how to reduce a graph using the GSPBox. Then we apply the pyramid to simple signal.
To start open a python shell (IPython is recommended here) and import the required packages. You would probably also import numpy as you will need it to create matrices and arrays.

>>> import numpy as np
>>> from pygsp.graphs import Sensor, compute_fourier_basis, estimate_lmax
>>> from pygsp.operators import kron_pyramid, pyramid_cell2coeff, pyramid_analysis, pyramid_synthesis

For this demo we will be using a Sensor graph with 512 nodes.

>>> G = Sensor(512, distribute=True)

The function kron_pyramid computes the graph pyramid for you:

>>> Gs = kron_pyramid(G, 5, epsilon=0.1, sparsify=False)

Next, we will compute the fourier basis of our different graph layers:
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
