*****
GSP Graph TV Demo - Reconstruction of missing sample on a graph using TV
*****

Description
########

Reconstruction of missing sample on a graph using TV

In this demo, we try to reconstruct missing sample of a piece-wise smooth signal on a graph. To do so, we will minimize the well-known TV norm defined on the graph.

For this example, you need the pyunlocbox. You can download it: https://github.com/epfl-lts2/pyunlocbox

We express the recovery problem as a convex optimization problem of the following form:

.. math:: arg \min_x  \|\nabla(x)\|_1 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

Where b represents the known measurements, M is an operator representing the mask and \epsilon is the radius of the l2 ball.

We set

* $f_1(x)=||\nabla x ||_1$
We define the prox of $f_1$ as:

.. math:: prox_{f1,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2 +  \gamma \| \nabla z \|_1

* $f_2$ is the indicator function of the set S define by $||Mx-b||_2 < \epsilon$
We define the prox of $f_2$ as

.. math:: prox_{f2,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2   + i_S(x)

with $i_S(x)$ is zero if x is in the set S and infinity otherwise.
This previous problem has an identical solution as:

.. math:: arg \min_{z} \|x - z\|_2^2   \hspace{1cm} such \hspace{0.25cm} that \hspace{1cm} \|Mz-b\|_2 \leq \epsilon

It is simply a projection on the B2-ball.

Results
########

.. image:: img/signal_graph.*

Original signal on graph

This figure shows the original signal on graph.

.. image:: img/depleted_signal.*

Depleted signal on graph

This figure shows the signal on graph after the application of the
mask and addition of noise. Half of the vertices are set to 0.

.. image:: img/tv_recons_signal.*

Reconstructed signal on graph using TV

This figure shows the reconstructed signal thanks to the algorithm.


Comparison with Tikhonov regularization
########

We can also use the Tikhonov regularizer that will promote smoothness.
In this case, we solve:

.. math:: arg \min_x \tau \|\nabla(x)\|_2^2 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon

The result is presented in the following figure:

.. image:: img/tv_recons_signal.*

Reconstructed signal on graph using Tikhonov

This figure shows the reconstructed signal thanks to the algorithm.
