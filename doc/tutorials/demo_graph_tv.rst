************************************************************************
GSP Graph TV Demo - Reconstruction of missing sample on a graph using TV
************************************************************************

Description
###########

Reconstruction of missing sample on a graph using TV

In this demo, we try to reconstruct missing sample of a piece-wise smooth signal on a graph. To do so, we will minimize the well-known TV norm defined on the graph.

For this example, you need the pyunlocbox. You can download it from https://github.com/epfl-lts2/pyunlocbox and installing it.

We express the recovery problem as a convex optimization problem of the following form:

.. math:: arg \min_x  \|\nabla(x)\|_1 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

Where :math:`b` represents the known measurements, :math:`M` is an operator representing the mask and :math:`\epsilon` is the radius of the l2 ball.

We set:

* :math:`f_1(x)=||\nabla x ||_1`

We define the prox of :math:`f_1` as:

.. math:: prox_{f1,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2 +  \gamma \| \nabla z \|_1

* :math:`f_2` is the indicator function of the set S define by $||Mx-b||_2 < \epsilon

We define the prox of :math:`f_2` as

.. math:: prox_{f2,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2   + i_S(x)

with :math:`i_S(x)` is zero if :math:`x` is in the set :math:`S` and infinity otherwise.
This previous problem has an identical solution as:

.. math:: arg \min_{z} \|x - z\|_2^2   \hspace{1cm} such \hspace{0.25cm} that \hspace{1cm} \|Mz-b\|_2 \leq \epsilon

It is simply a projection on the B2-ball.

Results and code
################

>>> import pygsp
>>> from pygsp import graphs, filters, operators, plotting
>>> import numpy as np
>>>
>>> # Create a random sensor graph
>>> G = graphs.Sensor(N=256, distribute=True)
>>> operators.compute_fourier_basis(G)
>>>
>>> # Create signal
>>> graph_value = np.copysign(np.ones(np.shape(G.U[:, 3])[0]), G.U[:, 3])
>>>
>>> plotting.plot_signal(G, graph_value, savefig=True, plot_name='doc/tutorials/img/original_signal')

.. figure:: img/original_signal.*

This figure shows the original signal on graph.

>>> # Create the mask
>>> M = np.random.rand(G.U.shape[0], 1)
>>> M = M > 0.6  # Probability of having no label on a vertex.
>>>
>>> # Applying the mask to the data
>>> sigma = 0.0
>>> depleted_graph_value = M * (graph_value.reshape(graph_value.size, 1) + sigma * np.random.standard_normal((G.N, 1)))
>>>
>>> plotting.plot_signal(G, depleted_graph_value, show_edges=True, savefig=True, plot_name='doc/tutorials/img/depleted_signal')

.. figure:: img/depleted_signal.*

This figure shows the signal on graph after the application of the
mask and addition of noise. More than half of the vertices are set to 0.

>>> # Setting the function f1 (see pyunlocbox for help)
>>> import pyunlocbox
>>> import math
>>>
>>> epsilon = sigma * math.sqrt(np.sum(M[:]))
>>> operatorA = lambda x: A * x
>>> f1 = pyunlocbox.functions.proj_b2(y=depleted_graph_value, A=operatorA, At=operatorA, tight=True, epsilon=epsilon)
>>>
>>> # Setting the function ftv
>>> f2 = pyunlocbox.functions.func()
>>> f1._prox = lambda x, T: operators.prox_tv(x, T, G, verbose=verbose-1)
>>> f1._eval = lambda x: operators.norm_tv(G, x)
>>>
>>> # Solve the problem
>>> solver = pyunlocbox.solvers.douglas_rachford()
>>> param = {'x0': depleted_graph_value, 'solver': solver, 'atol': 1e-7, 'maxit': 50, 'verbosity': 'LOW'}
>>> # With prox_tv
>>> ret = pyunlocboxsolvers.solve([f2, f1], **param)
>>> prox_tv_reconstructed_graph = ret['sol']
>>>
>>> plotting.plot_signal(G, prox_tv_reconstructed_graph, show_edges=True, savefig=True, plot_name='doc/tutorials/img/tv_recons_signal')

.. figure:: img/tv_recons_signal.*

This figure shows the reconstructed signal thanks to the algorithm.


Comparison with Tikhonov regularization
#######################################

We can also use the Tikhonov regularizer that will promote smoothness.
In this case, we solve:

.. math:: arg \min_x \tau \|\nabla(x)\|_2^2 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

The result is presented as following:

>>> # Solve the problem with the same solver as before but with a prox_tik function
>>> ret2 = pyunlocbox.solvers.solve([f3, f1], **param)
>>> prox_tik_reconstructed_graph = ret['sol']
>>>
>>> plotting.plot_signal(G, prox_tik_reconstructed_graph, show_edges=True, savefig=True, plot_name='doc/tutorials/img/tik_recons_signal')

.. figure:: img/tik_recons_signal.*

This figure shows the reconstructed signal thanks to the algorithm.
