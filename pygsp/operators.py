import numpy as np
import scipy as sp
from math import pi
from scipy import sparse
from scipy import linalg

from pygsp import utils


class operators(object):
    pass


def adj2vec(G):
    r"""
    Prepare the graph for the gradient computation

    Parameters
    ----------
    G : Graph structure

    Returns
    -------
    G : Graph structure
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
    Parameters
    ----------
    G : Graph structure
    s : Signal living on the nodes

    Returns
    -------
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

    Parameters
    ----------
    G : Graph structure
    s : Signal living on the nodes

    Returns
    -------
    gr : Gradient living on the edges

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

    Parameters
    ----------
    G : Graph structure

    Returns
    -------
    D : Gradient sparse matrix

    """
    if not hasattr(G, 'v_in'):
        G = adj2vec(G)
        print('To be more efficient you should run: G = adj2vec(G); \
              before using this proximal operator.')

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

    Parameters
    ----------
    G : Graph or Fourier basis
    f : f (signal)

    Returns
    -------
    f_hat : Graph Fourier transform of *f*
    """

    if isinstance(G, graphs.Graph):
        if not hasattr(G, 'U'):
            raise AttributeError('You need first to compute the Fourier basis.\
                                  You can do it with the function \
                                 compute_fourier_basis')

        else:
            U = G.U

    else:
        U = G

    return U.transpose().conjugate()*f


def gwft(G, g, f, verbose=1, lowmemory=True):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    g : Window (graph signal or kernel)
    f : Graph signal (column vector)
    verbose (int) : 0 no log, 1 print main steps, 2 print all steps
        Default is 1
    lowmemory (bool) : use less memory
        Default is True

    Returns
    -------
    C : Coefficient.
    """
    Nf = np.shape(f)[1]

    if not hasattr(G, 'U'):
        raise AttributeError('You need first to compute the Fourier basis. You can do it with the function compute_fourier_basis')

    # if iscell(g)
    #    g = gsp_igft(G,g{1}(G.e))

    if hasattr(g, 'function_handle'):
        g = gsp_igft(G, g(G.e))

    if not lowmemory:
        # Compute the Frame into a big matrix
        Frame = gwft_frame_matrix(G, g, verbose=verbose)

        C = Frame.transpose()*f
        C = C.reshape(G.N, G.N, Nf)

    else:
        # Compute the translate of g
        ghat = G.U.transpose()*g
        Ftrans = np.sqrt(G.N)*G.U*(np.kron(np.ones((G.N)), ghat)*G.U.transpose())
        C = zeros((G.N, G.N))

        for jj in range(1, Nf):
            for ii in range(1, G.N):
                C[:, ii, jj] = (np.kron(np.ones((G.N)), 1./G.U[:, 1])*G.U*np.kron(np.ones((G.N)), Ftrans[:, ii])).transpose()*f[:, jj]

    return C


def gwft2(G, f, k, verbose=1):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    f : Graph signal
    k : kernel
    param : Structure of optional parameter

    Returns
    -------
    C : Coefficient.
    """
    if not hasattr(G, 'E'):
        raise ValueError('You need first to compute the Fourier basis .You can do it with the function compute_fourier_basis.')

    g = filters.gabor_filterbank(G, k)

    C = filters.analysis(G, g, f, verbose=verbose)
    C = transpose(vec2mat(C, G.N))

    return C


def gwft_frame_matrix(G, g, param):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
    G : Graph
    g : window
    param : Structure of optional parameter

    Returns
    -------
        F : Frame
    """
    raise NotImplementedError
    return F


def igth(G, f_hat):
    r"""
    Inverse graph Fourier transform

    Parameters
    ----------
    G : Graph or Fourier basis
    f_hat : Signal

    Returns
    -------
    f : Inverse graph Fourier transform of *f_hat*

    """
    if isinstance(G, graphs.Graph):
        if not hasattr(G, 'U'):
            raise AttributeError('You need first to compute the Fourier basis.\
                                  You can do it with the function \
                                 compute_fourier_basis')

        else:
            U = G.U

    else:
        U = G

    return f_hat*U


def ngwft(G, f, g, param):
    r"""
    Normalized graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    f : Graph signal
    g : window
    param : Structure of optional parameter

    Returns
    -------
    C : Coefficient
    """
    raise NotImplementedError
    return C


def ngwft_frame_matrix(G, g, param):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
    G : Graph
    g : window
    param : Structure of optional parameter

    Output parameters:
    F : Frame
    """
    raise NotImplementedError

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


@utils.filterbank_handler
def compute_cheby_coeff(f, G, m=30, N=None, i=0, *args):

    if not N:
        N = m + 1

    if not hasattr(G, 'lmax'):
        G.lmax = utils.estimate_lmax(G)
        print('The variable lmax has not been computed yet, it will be done \
              but if you have to compute multiple times you can precompute \
              it with pygsp.utils.estimate_lmax(G)')
    print(G.lmax)
    a_arange = range(0, int(G.lmax))

    a1 = (a_arange[2]-a_arange[1])/2
    a2 = (a_arange[2]+a_arange[1])/2
    c = np.zeros(m+1)

    for o in range(m+1):
        c[o] = np.sum(f.g[i](a1 * np.cos(pi * (np.arange(1, N)-0.5))/N) + a2 *
                      np.cos(pi * (o-1) * (np.arange(1, N)-0.5)/N)) * 2/N
    return c


def full_eigen(L):
    eigenvectors, eigenvalues, _ = np.linalg.svd(L.todense())

    # Sort everything

    inds = np.argsort(eigenvalues)
    EVa = np.sort(eigenvalues)

    # TODO check if axis are good
    EVe = eigenvectors[:, inds]

    for val in EVe[0, :].reshape(EVe.shape[0], 1):
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

    Parameters
    ----------
    G : Graph
    g : kernel (or filterbank)
    i : Indices of vertex (int)

    Returns
    -------
    gt : translate signal
    """
    raise NotImplementedError

    f = np.zeros((G.N))
    f[i-1] = 1

    gt = sqrt(G.N)*filters.filters_analysis(G, g, f)

    return gt


def kron_pyramid(G, Nlevels, param):
    r"""
    Compute a pyramid of graphs using the kron reduction

    Parameters
    ----------
    G : Graph structure
    Nlevels : Number of level of decomposition
    param : Optional structure of parameters

    Returns
    -------
    Cs : Cell array of graphs
    """
    raise NotImplementedError

    return Gs


def gsp_kron_reduction(G, ind):
    r"""
    Compute the kron reduction

    Parameters
    ----------
    G : Graph structure or weight matrix
    ind : indices of the nodes to keep

    Returns
    -------
    Gnew : New graph structure or weight matrix
    """
    raise NotImplementedError

    return Gnew


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid

    Parameters
    ----------
    ca : Cell array with the coarse approximation at each level
    pe : Cell array with the prediction errors at each level

    Returns
    -------
    coeff : Vector of coefficient
    """
    raise NotImplementedError

    return coeff


def pyramid_synthesis(Gs, coeff, param):
    r"""
    Synthesizes a signal from its graph pyramid transform coefficients

    Parameters
    ----------
    Gs : A multiresolution sequence of graph structures.
    coeff : The coefficients to perform the reconstruction

    Returns
    -------
    signal : The synthesized signal.
    ca : Cell array with the coarse approximation at each level
    """
    raise NotImplementedError

    return [signal, ca]


def modulate(G, f, k):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
    G : Graph
    f : Signal (column)
    k : Indices of frequencies (int)

    Returns
    -------
    fm : Modulated signal
    """
    raise NotImplementedError

    return fm


def translate(G, f, i):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
    G : Graph
    f : Signal (column)
    i : Indices of vertex (int)

    Returns
    -------
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
    Nlevel : Number of times to downsample and coarsen the tree
    root : The index of the root of the tree (default=1)
    reduction_method : The graph reduction method (default='resistance_distance')
    compute_full_eigen : To also compute the graph Laplacian eigenvalues for every tree in the sequence

    Returns
    -------
    Gs : Cell array, with each element containing a graph structure represent a reduced tree.
    subsampled_vertex_indices : Indices of the vertices of the previous tree that are kept for the subsequent tree.
    """
    raise NotImplementedError

    return [Gs, subsampled_vertex_indices]
