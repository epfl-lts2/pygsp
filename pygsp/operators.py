import numpy as np

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

    if hasattr(G, e) or hasattr(G, U):
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
        G.U, G.e = full_eigen(G.L)


def full_eigen(L):
    eigenvalues, eigenvectors = np.linalg.svd(L)
