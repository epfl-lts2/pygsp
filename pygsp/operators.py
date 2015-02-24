import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg
from pygsp import utils


class operators(object):
    pass


def adj2vec(G):
    r"""
    Prepare the graph for the gradient computation

    Input parameters:
        G   : Graph structure
    Output parameters:
        G   : Graph structure
    """
    if G.directed:
        raise NotImplementedError("Not implemented yet")

    else:
        v_i, v_j = (sparse.tril(G.W)).nonzero()
        weights = G.W[v_i, v_j]

        # TODO G.ind_edges = sub2ind(size(G.W), G.v_in, G.v_out)
        G.v_in = v_i
        G.v_out = v_j
        G.weights = weights
        G.Ne = np.shape(v_i)[0]

        G.Diff = grad_mat(G)


def div(G, s):
    r"""
    """
    if hasattr(G, 'lap_type'):
        if G.lap_type == 'combinatorial':
            raise NotImplementedError('Not implemented yet. However ask Nathanael it is very easy')

    if G.Ne != np.shape(s)[0]:
        raise ValueError('Signal size not equal to number of edges')

    D = grad_mat(G)
    di = D.getH()*s

    if s.dtype == 'float32':
        di = np.float32(di)

    return di


def gft(G, f):
    r"""
    Graph Fourier transform
    Usage:  f_hat=gsp_gft(G,f);

    Input parameters:
          G          : Graph or Fourier basis
          f          : f (signal)
    Output parameters:
          f_hat      : Graph Fourier transform of *f*
    """

    if isinstance(G, pygsp.graphs.Graph):
        if hasattr(G, 'U'):
            raise AttributeError('You need first to compute the Fourier basis. You can do it with the function compute_fourier_basis')

        else:
            U = G.U

    else:
        U = G

    return U.transpose().conjugate()*f


def grad(G, s):
    r"""
    Graph gradient
    Usage: gr = gsp_grad(G,s)

    Input parameters:
        G   : Graph structure
       s   : Signal living on the nodes
    Output parameters:
        gr  : Gradient living on the edges

    """
    if hasattr(G, 'lap_type'):
        if G.lap_type == 'combinatorial':
            raise NotImplementedError('Not implemented yet. However ask Nathanael it is very easy')

    D = grad_mat(G)
    gr = D*s

    if s.dtype == 'float32':
        gr = np.float32(gr)

    return gr


def grad_mat(G):
    r"""
    Gradient sparse matrix of the graph G
    Usage:  D = gsp_gradient_mat(G);

    Input parameters:
        G   : Graph structure

    Output parameters:
        D   : Gradient sparse matrix

    """
    if not hasattr(G, 'v_in'):
        G = adj2vec(G)
        print('To be more efficient you should run: G = adj2vec(G); before using this proximal operator.')

    if hasattr(G, 'Diff'):
        D = G.Diff

    else:
        n = G.Ne
        Dc = np.ones((2*n))
        Dv = np.ones((2*n))

        Dr = np.concatenate((np.arange(n), np.arange(n)))
        Dc[:n] = G.v_in
        Dc[n:] = G.v_out
        Dv[:n] = np.sqrt(G.weights)
        Dv[n:] = -np.sqrt(G.weight)
        D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, G.N))

    return D


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
        if not hasattr(G, 'L'):
            raise AttributeError("Graph Laplacian is missing")
        G.e, G.U = full_eigen(G.L)

    G.lmax = np.max(G.e)

    G.mu = np.max(np.abs(G.U))


def compute_cheby_coeff(G, f, m=30, N=None, *args):

    if not N:
        N = m + 1

    if isinstance(f, list):
        Nf = len(f)
        c = np.zeros(m+1, Nf)


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
