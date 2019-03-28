r"""
Concentration of the eigenvalues
================================

The eigenvalues of the graph Laplacian concentrates to the same value as the
graph becomes full.
"""

from matplotlib import pyplot as plt
import pygsp as pg

n_neighbors = [1, 2, 5, 8]
fig, axes = plt.subplots(4, len(n_neighbors), figsize=(15, 10))

for k, ax in zip(n_neighbors, axes.T):
    graph = pg.graphs.Ring(17, k=k)
    graph.compute_fourier_basis()
    graph.plot(graph.U[:, 1], ax=ax[0])
    ax[0].axis('equal')
    ax[1].spy(graph.W)
    ax[2].plot(graph.e, '.')
    ax[2].set_title('k={}'.format(k))
    graph.set_coordinates('line1D')
    graph.plot(graph.U[:, :4], ax=ax[3], title='')

fig.tight_layout()
