===============================================
Reconstruction of missing samples with graph TV
===============================================

.. note::
    The toolbox is **not ready** (yet?) for the completion of that tutorial.
    For one, the proximal TV operator on graph is missing.
    Please see the `matlab version of that tutorial
    <https://lts2.epfl.ch/gsp/doc/demos/gsp_demo_graph_tv.php>`_.
    If you like it, implement it!

Description
-----------

In this demo, we try to reconstruct missing sample of a piece-wise smooth signal on a graph. To do so, we will minimize the well-known TV norm defined on the graph.

For this example, you will need the `pyunlocbox <https://github.com/epfl-lts2/pyunlocbox>`_.

We express the recovery problem as a convex optimization problem of the following form:

.. math:: arg \min_x  \|\nabla(x)\|_1 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

Where :math:`b` represents the known measurements, :math:`M` is an operator representing the mask and :math:`\epsilon` is the radius of the l2 ball.

We set:

* :math:`f_1(x)=||\nabla x ||_1`

We define the prox of :math:`f_1` as:

.. math:: prox_{f1,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2 +  \gamma \| \nabla z \|_1

* :math:`f_2` is the indicator function of the set S define by :math:||Mx-b||_2 < \epsilon

We define the prox of :math:`f_2` as

.. math:: prox_{f2,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2   + i_S(x)

with :math:`i_S(x)` is zero if :math:`x` is in the set :math:`S` and infinity otherwise.
This previous problem has an identical solution as:

.. math:: arg \min_{z} \|x - z\|_2^2   \hspace{1cm} such \hspace{0.25cm} that \hspace{1cm} \|Mz-b\|_2 \leq \epsilon

It is simply a projection on the B2-ball.

Results and code
----------------

.. plot::
    :context: reset

    >>> import numpy as np
    >>> from pygsp import graphs, plotting
    >>> plotting.BACKEND = 'matplotlib'
    >>>
    >>> # Create a random sensor graph
    >>> G = graphs.Sensor(N=256, distribute=True)
    >>> G.compute_fourier_basis()
    >>>
    >>> # Create signal
    >>> graph_value = np.copysign(np.ones(np.shape(G.U[:, 3])[0]), G.U[:, 3])
    >>>
    >>> G.plot_signal(graph_value)

This figure shows the original signal on graph.

.. plot::
    :context: close-figs

    >>> # Create the mask
    >>> M = np.random.rand(G.U.shape[0], 1)
    >>> M = M > 0.6  # Probability of having no label on a vertex.
    >>>
    >>> # Applying the mask to the data
    >>> sigma = 0.0
    >>> depleted_graph_value = M * (graph_value.reshape(graph_value.size, 1) + sigma * np.random.standard_normal((G.N, 1)))
    >>>
    >>> G.plot_signal(depleted_graph_value, show_edges=True)

This figure shows the signal on graph after the application of the
mask and addition of noise. More than half of the vertices are set to 0.

.. plot::
    :context: close-figs

    >>> # Setting the function f1 (see pyunlocbox for help)
    >>> # import pyunlocbox
    >>> # import math
    >>>
    >>> # epsilon = sigma * math.sqrt(np.sum(M[:]))
    >>> # operatorA = lambda x: A * x
    >>> # f1 = pyunlocbox.functions.proj_b2(y=depleted_graph_value, A=operatorA, At=operatorA, tight=True, epsilon=epsilon)
    >>>
    >>> # Setting the function ftv
    >>> # f2 = pyunlocbox.functions.func()
    >>> # f2._prox = lambda x, T: operators.prox_tv(x, T, G, verbose=verbose-1)
    >>> # f2._eval = lambda x: operators.norm_tv(G, x)
    >>>
    >>> # Solve the problem with prox_tv
    >>> # ret = pyunlocbox.solvers.solve(
    >>> #         [f2, f1],
    >>> #         x0=depleted_graph_value,
    >>> #         solver=pyunlocbox.solvers.douglas_rachford(),
    >>> #         atol=1e-7,
    >>> #         maxit=50,
    >>> #         verbosity='LOW')
    >>> # prox_tv_reconstructed_graph = ret['sol']
    >>>
    >>> # G.plot_signal(prox_tv_reconstructed_graph, show_edges=True)

This figure shows the reconstructed signal thanks to the algorithm.

Comparison with Tikhonov regularization
---------------------------------------

We can also use the Tikhonov regularizer that will promote smoothness.
In this case, we solve:

.. math:: arg \min_x \tau \|\nabla(x)\|_2^2 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

The result is presented as following:

.. plot::
    :context: close-figs

    >>> # Solve the problem with the same solver as before but with a prox_tik function
    >>> # ret = pyunlocbox.solvers.solve(
    >>> #         [f3, f1],
    >>> #         x0=depleted_graph_value,
    >>> #         solver=pyunlocbox.solvers.douglas_rachford(),
    >>> #         atol=1e-7,
    >>> #         maxit=50,
    >>> #         verbosity='LOW')
    >>> # prox_tik_reconstructed_graph = ret['sol']
    >>>
    >>> # G.plot_signal(prox_tik_reconstructed_graph, show_edges=True)

This figure shows the reconstructed signal thanks to the algorithm.
