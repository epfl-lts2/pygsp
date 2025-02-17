r"""
Graph structure captured by the Fourier modes
=============================================

As an eigendecomposition of the graph Laplacian :attr:`pygsp.graphs.Graph.L`,
the Fourier modes :attr:`pygsp.graphs.Graph.U` capture the structure of the
graph. Similarly to
`PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
(which is an eigendecomposition of a covariance matrix), the larger-scale
structure is captured by the first modes (ordered by eigenvalue). The later
ones capture more and more details. That is why a signal that is well explained
by a graph has a low-frequency content.

`Laplacian eigenmaps <https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Laplacian_eigenmaps>`_
use this for dimensionality reduction.
"""

import numpy as np
from scipy import sparse
import pygsp as pg
from matplotlib import pyplot as plt

N_VERTICES = 100

def error(L, U, e):
    normalization = sparse.linalg.norm(L, ord='fro')
    return np.linalg.norm(L - U @ np.diag(e) @ U.T, ord='fro') / normalization

def errors(graph):
    graph.compute_fourier_basis()
    return [error(graph.L, graph.U[:, :k], graph.e[:k]) for k in range(N_VERTICES+1)]

graphs = [
    pg.graphs.FullConnected(N_VERTICES),
    pg.graphs.StochasticBlockModel(N_VERTICES),
    pg.graphs.Sensor(N_VERTICES),
    pg.graphs.ErdosRenyi(N_VERTICES),
    pg.graphs.Grid2d(int(N_VERTICES**0.5)),
    pg.graphs.Community(N_VERTICES),
    pg.graphs.Comet(N_VERTICES),
    pg.graphs.SwissRoll(N_VERTICES),
    pg.graphs.BarabasiAlbert(N_VERTICES),
]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for graph in graphs:
    ax.plot(errors(graph), '-', label=graph.__class__.__name__)
ax.set_xlabel('number of Fourier modes $k$')
ax.set_ylabel('reconstruction error');
ax.set_title(r'Laplacian reconstruction error $\frac{\| L - U_k \Lambda_k U_k^\top \| }{\|L\|}$', fontsize=16)
ax.legend(loc='lower left')
