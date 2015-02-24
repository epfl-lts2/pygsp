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
        if not hasattr(G, 'U'):
            raise AttributeError('You need first to compute the Fourier basis. You can do it with the function compute_fourier_basis')

        else:
            U = G.U

    else:
        U = G

    return U.transpose().conjugate()*f


def gwft(G, g, f, param):
    r"""
    Graph windowed Fourier transform

    Input parameters:
          G     : Graph
          g     : Window (graph signal or kernel)
          f     : Graph signal (column vector)
          param : Structure of optional parameter

    Output parameters:
          C     : Coefficient.
    """

    return C


def gwft2(G, f, k, param):
    r"""
    Graph windowed Fourier transform

    Input parameters:
          G     : Graph
          f     : Graph signal
          k     : kernel
          param : Structure of optional parameter

    Output parameters:
          C     : Coefficient.
    """

    return C


def gwft_frame_matrix(G, g, param):
    r"""
    Create the matrix of the GWFT frame

    Input parameters:
          G     : Graph
          g     : window
          param : Structure of optional parameter

    Output parameters:
          F     : Frame
    """

    return F


def igth(G, f_hat):
    r"""
    Inverse graph Fourier transform

    Input parameters:
          G          : Graph or Fourier basis
          f_hat      : Signal

    Output parameters:
          f          : Inverse graph Fourier transform of *f_hat*

    """
    if isinstance(G, pygsp.graphs.Graph):
        if not hasattr(G, 'U'):
            raise AttributeError('You need first to compute the Fourier basis. You can do it with the function compute_fourier_basis')

        else:
            U = G.U

    else:
        U = G

    return f_hat*U


def ngwft(G, f, g, param):
    r"""
    Normalized graph windowed Fourier transform

    Input parameters:
          G     : Graph
          f     : Graph signal
          g     : window
          param : Structure of optional parameter

    Output parameters:
          C     : Coefficient
    """

    return C


def ngwft_frame_matrix(G, g, param):
    r"""
    Create the matrix of the GWFT frame

    Input parameters:
          G     : Graph
          g     : window
          param : Structure of optional parameter

    Output parameters:
          F     : Frame
    """

    return F

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


def create_laplacian(G):
    if sp.shape(G.W) == (1, 1):
        return sparse.lil_matrix(0)
    else:
        if G.lap_type == 'combinatorial':
            L = sparse.lil_matrix(np.diagflat(G.W.sum(1)) - G.W)
        elif G.lap_type == 'normalized':
            D = sparse.lil_matrix(G.W.sum(1).diagonal() ** (-0.5))
            L = sparse.lil_matrix(np.matlib.identity(G.N)) - D * G.W * D
        elif G.lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')
        return L


def localize(G, g, i):
    r"""
    Localize a kernel g to the node i

    Input parameters
        G   : Graph
        g   : kernel (or filterbank)
        i   : Indices of vertex (int)

    Output parameters
        gt  : translate signal
    """

    return gt


def kron_pyramid(G, Nlevels, param):
    r"""
    Compute a pyramid of graphs using the kron reduction

    Input parameters:
        G       : Graph structure
        Nlevels : Number of level of decomposition
        param   : Optional structure of parameters

    Output parameters:
        Gs      : Cell array of graphs
    """

    return Gs


def gsp_kron_reduction(G, ind):
    r"""
    Compute the kron reduction

    Input parameters:
        G       : Graph structure or weight matrix
        ind     : indices of the nodes to keep

    Output parameters:
        Gnew    : New graph structure or weight matrix
    """

    return Gnew


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid

    Input parameters:
        ca      : Cell array with the coarse approximation at each level
        pe      : Cell array with the prediction errors at each level

    Output parameters:
       coeff   : Vector of coefficient
    """

    return coeff


def pyramid_synthesis(Gs, coeff, param):
    r"""
    Synthesizes a signal from its graph pyramid transform coefficients

    Input parameters:
        Gs      : A multiresolution sequence of graph structures.
        coeff   : The coefficients to perform the reconstruction

    Output parameters:
        signal  : The synthesized signal.
        ca      : Cell array with the coarse approximation at each level
    """

    return [signal, ca]


def modulate(G, f, k):
    r"""
    Tranlate the signal f to the node i

    Input parameters
        G   : Graph
        f   : Signal (column)
        k   : Indices of frequencies (int)
    Output parameters
        fm  : Modulated signal
    """

    return fm


def translate(G, f, i):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
        G : Graph
        f : Signal (column)
        i : Indices of vertex (int)

    Output
    ------
        ft : translate signal
    """

    fhat = gft(G, f)
    nt = np.shape(f)[1]

    ft = np.sqrt(G.N)*igft(G, fhat, np.kron(np.ones((1, nt)), G.U[i]))

    return ft


def tree_multiresolution(G, Nlevel, param):
    r"""
    Compute a multiresolution of trees

    Parameters
    ----------
          G : Graph structure of a tree.
          Nlevel : Number of times to downsample and coarsen the tree.

    Output
    ------
          Gs : Cell array, with each element containing a graph structure represent a reduced tree.
          subsampled_vertex_indices : Indices of the vertices of the previous tree that are kept for the subsequent tree.

    Additional parameters
    ---------------------
          param.root : The index of the root of the tree (default=1)
          param.reduction_method : The graph reduction method (default='resistance_distance')
          param.compute_full_eigen : To also compute the graph Laplacian eigenvalues for every tree in the sequence
    """

    return [Gs, subsampled_vertex_indices]
