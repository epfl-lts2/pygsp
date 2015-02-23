import numpy as np
import scipy as sp
from scipy import linalg

from pygsp import utils


class operators(object):
    pass


def grad(G, s):
    r"""
    Graph gradient

    """
    pass


def grad_mat(G):
    r"""
    Gradient sparse matrix of the graph

    """
    pass


@utils.graph_array_handler
def compute_fourier_basis(G, exact=None, cheb_order=30, **kwargs):

    if hasattr(G, 'e') or hasattr(G, 'U'):
        print("This graph already has Laplacian eigenvectors or eigenvalues")

    if G.N > 3000:
        print("Performing full eigendecomposition of a large matrix\
              may take some time.")

    if False:
        # TODO
        pass
    else:
        if not hasattr(G, L):
            raise AttributeError("Graph Laplacian is missing")
        G.e, G.U = full_eigen(G.L)

    G.lmax = np.max(G.e)

    G.mu = np.max(np.abs(G.U))


def full_eigen(L):
    eigenvectors, eigenvalues, _ = np.linalg.svd(L.todense())

    # Sort everything
    inds = np.argsort(eigenvalues)
    EVa = np.sort(eigenvalues)

    # TODO check if axis are good
    EVe = eigenvectors[:, inds]

    for val in EVe[0, :]:
        if val < 0:
            val = -val

    return EVa, EVe
