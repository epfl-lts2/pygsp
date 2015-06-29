# -*- coding: utf-8 -*-
r"""
This module implements the main operators for the PyGSP box.
"""

import numpy as np
import scipy as sp
from math import pi, sqrt
from scipy import sparse
from scipy import linalg

from pygsp import utils


logger = utils.build_logger(__name__)

def adj2vec(G):
    r"""
    Prepare the graph for the gradient computation

    Parameters
    ----------
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
    s : ndarray
        Signal living on the nodes

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
    s : ndarray
        Signal living on the nodes

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

    Parameters
    ----------
    G : Graph or Fourier basis
    f : ndarray
        must be in 2d, even if the second dim is 1 signal

    Returns
    -------
    f_hat : ndarray
        Graph Fourier transform of *f*
    """

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if not hasattr(G, 'U'):
            logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
            compute_fourier_basis(G)

        U = G.U
    else:
        U = G

    return np.dot(np.conjugate(U.T), f)


def gwft(G, g, f, lowmemory=True):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    g : ndarray or Filter
        Window (graph signal or kernel)
    f : ndarray
        Graph signal
    lowmemory : bool
        use less memory
        Default is True

    Returns
    -------
    C : ndarray
        Coefficients

    """
    Nf = np.shape(f)[1]

    if not hasattr(G, 'U'):
        logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
        compute_fourier_basis(G)

    # if iscell(g)
    #    g = gsp_igft(G,g{1}(G.e))

    if hasattr(g, 'function_handle'):
        g = gsp_igft(G, g.g[0](G.e))

    if not lowmemory:
        # Compute the Frame into a big matrix
        Frame = gwft_frame_matrix(G, g)

        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N, Nf), order='F')

    else:
        # Compute the translate of g
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))
        C = zeros((G.N, G.N))

        for j in range(Nf):
            for i in range(G.N):
                C[:, i, j] = (np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.dot(np.kron(np.ones((G.N)), Ftrans[:, i])).T, f[:, j])

    return C


def gwft2(G, f, k):
    r"""
    Graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    f : ndarray
        Graph signal
    k : #TODO
        kernel

    Returns
    -------
    C : Coefficient.
    """
    from pygsp.filters import analysis, gabor_filterbank

    if not hasattr(G, 'e'):
        logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
        compute_fourier_basis(G)

    g = gabor_filterbank(G, k)

    C = analysis(G, g, f)
    C = utils.vec2mat(C, G.N).T

    return C


def gwft_frame_matrix(G, g):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
    G : Graph
    g : window

    Returns
    -------
        F : TODO
            Frame
    """
    if G.N > 256:
        logger.warning("It will create a big matrix. You can use other methods.")

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))

    F = utils.repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    return F


def igft(G, f_hat):
    r"""
    Inverse graph Fourier transform

    Parameters
    ----------
    G : Graph or Fourier basis
    f_hat : ndarray
        Signal

    Returns
    -------
    f : Inverse graph Fourier transform of *f_hat*

    """

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if not hasattr(G, 'U'):
            logger.info('analysis filter has to compute the eigenvalues and the eigenvectors.')
            compute_fourier_basis(G)
        U = G.U

    else:
        U = G

    return np.dot(U, f_hat)


def ngwft(G, f, g, lowmemory=True):
    r"""
    Normalized graph windowed Fourier transform

    Parameters
    ----------
    G : Graph
    f : ndarray
        Graph signal
    g : TODO
        window
    lowmemory : bool
        use less memory. (default = True)

    Returns
    -------
    C : ndarray
        Coefficients
    """

    if not hasattr(G, 'U'):
        logger.ingo('analysis filter has to compute the eigenvalues and the eigenvectors.')
        compute_fourier_basis(G)

    if lowmemory:
        # Compute the Frame into a big matrix
        Frame = ngwft_frame_matrix(G, g)
        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N), order='F')

    else:
        # Compute the translate of g
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))
        C = np.zeros((G.N, G.N))

        for i in range(G.N):
            atoms = np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.kron(np.ones((G.N)), Ftrans[:, i]).T

            # normalization
            atoms /= np.kron((np.ones((G.N))), np.sqrt(np.sum(np.abs(atoms),
                                                              axis=0)))
            C[:, i] = np.dot(atoms, f)

    return C


def ngwft_frame_matrix(G, g):
    r"""
    Create the matrix of the GWFT frame

    Parameters
    ----------
    G : Graph
    g : TODO
        window

    Output parameters:
    F : TODO
        Frame
    """
    if G.N > 256:
        logger.warning('It will create a big matrix, you can use other methods.')

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(g.N)*np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))

    F = repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    # Normalization
    F /= np.kron((np.ones((G.N)), np.sqrt(np.sum(np.power(np.abs(F), 2),
                                          axiis=0))))

    return F


@utils.graph_array_handler
def compute_fourier_basis(G, exact=None, cheb_order=30, **kwargs):
    r"""
    TODO
    """

    if hasattr(G, 'e') or hasattr(G, 'U'):
        print("This graph already has Laplacian eigenvectors or eigenvalues")
        return

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
        G.e = np.array(G.e)
        G.U = np.array(G.U)

    G.lmax = np.max(G.e)

    G.mu = np.max(np.abs(G.U))


@utils.filterbank_handler
def compute_cheby_coeff(f, G=None, m=30, N=None, i=0, *args):
    r"""
    Compute Chebyshev coefficients for a Filterbank

    Paramters
    ---------
    f : Filter or list of filters
    G : Graph
    m : int
        Maximum order of Chebyshev coeff to compute (default = 30)
    N : int
        Grid order used to compute quadrature (default = m + 1)
    i : int
        Indice of the Filterbank element to compute (default = 0)

    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients

    """

    if G is None:
        G = f.G

    if not N:
        N = m + 1

    if not hasattr(G, 'lmax'):
        G.lmax = utils.estimate_lmax(G)
        print('The variable lmax has not been computed yet, it will be done.)')

    a_arange = [0, G.lmax]

    a1 = (a_arange[1] - a_arange[0])/2
    a2 = (a_arange[1] + a_arange[0])/2
    c = np.zeros((m+1))

    for o in range(m+1):
        c[o] = np.sum(f.g[i](a1*np.cos(pi*(np.arange(N) + 0.5)/N) + a2)*np.cos(pi*o*(np.arange(N) + 0.5)/N)) * 2./N

    return c


def cheby_op(G, c, signal, **kwargs):
    r"""
    Chebyshev polylnomial of graph Laplacian applied to vector

    Parameters
    ----------
    G : Graph
    c : ndarray
        Chebyshev coefficients
    signal : ndarray
        Signal to filter

    Returns
    -------
    r : ndarray
        Result of the filtering

    """
    # With that way, we can handle if we do not have a list of filter but only a simple filter.
    if type(c) != list:
        c = [c]

    M = np.shape(c[0])[0]
    Nscales = len(c)

    try:
        M >= 2
    except:
        print("The signal has an invalid shape")

    if not hasattr(G, 'lmax'):
        G.lmax = utils.estimate_lmax(G)

    if signal.dtype == 'float32':
        signal = np.float64(signal)

    # thanks to that, we can also have 1d signal.
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N * Nscales, Nv))
    except IndexError:
        r = np.zeros((G.N * Nscales))

    a_arange = [0, G.lmax]

    a1 = float(a_arange[1]-a_arange[0])/2
    a2 = float(a_arange[1]+a_arange[0])/2

    twf_old = signal
    twf_cur = (np.dot(G.L.toarray(), signal) - a2 * signal)/a1

    for i in range(Nscales):
        r[np.arange(G.N) + G.N*i] = 0.5*c[i][0]*twf_old + c[i][1]*twf_cur
    for k in range(2, M+1):
        twf_new = (2./a1) * (np.dot(G.L.toarray(), twf_cur) - a2*twf_cur) - twf_old
        for i in range(Nscales):
            if k + 1 <= M:
                r[np.arange(G.N) + G.N*i] += c[i][k]*twf_new

        twf_old = twf_cur
        twf_cur = twf_new

    return r


def full_eigen(L):
    r"""
    Computes full eigen decomposition on a matrix

    Parameters
    ----------
    L : ndarray
        Matrix to decompose

    Returns
    -------
    EVa : ndarray
        Eigenvalues
    EVe : ndarray
        Eigenvectors

    """

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


@utils.graph_array_handler
def create_laplacian(G, lap_type=None, get_laplacian_only=True):
    r"""
    Create the graph laplacian of graph G

    Parameters
    ----------
    G : Graph
    lap_type : string :
        the laplacian type to use.
        Default is the lap_type attribute of G, otherwise it is "combinatorial".
    get_laplacian_only : bool
        True return each Laplacian in an array
        False set each Laplacian in each graphs.
        (default = True)

    Returns
    -------
    L : ndarray
        Laplacian matrix

    """
    if sp.shape(G.W) == (1, 1):
        return sparse.lil_matrix(0)

    if not lap_type:
        if not hasattr(G, 'lap_type'):
            lap_type = 'combinatorial'
            G.lap_type = lap_type
        else:
            lap_type = G.lap_type

    G.lap_type = lap_type

    if G.directed:
        if lap_type == 'combinatorial':
            L = 0.5*sparse.lil_matrix(np.diagflat(G.W.sum(0)) + np.diagflat(G.W.sum(1)) - G.W - G.W.getH())
        elif lap_type == 'normalized':
            raise NotImplementedError('Yet. Ask Nathanael.')
        elif lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')

    else:
        if lap_type == 'combinatorial':
            L = sparse.lil_matrix(np.diagflat(G.W.sum(1)) - G.W)
        elif lap_type == 'normalized':
            D = sparse.lil_matrix(np.diaflat(np.power(G.W.sum(1), -0.5)))
            L = sparse.identity(G.N) - D * G.W * D
        elif lap_type == 'none':
            L = sparse.lil_matrix(0)
        else:
            raise AttributeError('Unknown laplacian type!')

    if get_laplacian_only:
        return L
    else:
        G.L = L


def lanczos_op(fi, s, G=None, order=30):
    r"""
    Perform the lanczos approximation of the signal s

    Parameters
    ----------
    fi: Filter or list of filters
    s : ndarray
        Signal to approximate.
    G : Graph
    order : int
        Degree of the lanczos approximation. (default = 30)



    Returns
    -------
    L : ndarray
        lanczos approximation of s

    """
    if not G:
        G = fi.G

    Nf = len(fi.g)
    Nv = np.shape(s)[1]
    c = np.zeros((G.N))

    for j in range(Nv):
        V, H = lanczos(G.L, order, s[:, j])
        Uh, Eh = np.linalg.eig(H)
        V = np.dot(V, Uh)

        Eh = np.diagonal(Eh)
        Eh = np.where(Eh < 0, 0, Eh)
        fie = fi.evaluate(Eh)

        for i in range(Nf):
            c[np.range(G.N) + i*G.N, j] = np.dot(V, fie[:][i] * np.dot(V.T, s[:, j]))

    return c


def localize(G, g, i):
    r"""
    Localize a kernel g to the node i

    Parameters
    ----------
    G : Graph
    g : TODO
        kernel (or filterbank)
    i : int
        Indices of vertex

    Returns
    -------
    gt : translate signal

    """
    raise NotImplementedError

    f = np.zeros((G.N))
    f[i-1] = 1

    gt = sqrt(G.N) * filters_analysis(G, g, f)

    return gt


def kron_pyramid(G, Nlevels, lamda=0.025, sparsify=False, epsilon=None):
    r"""
    Compute a pyramid of graphs using the kron reduction

    Parameters
    ----------
    G : Graph structure
    Nlevels : int
        Number of level of decomposition
    lambda : float
        Stability parameter. It add self loop to the graph to give the alorithm some stability.
        (default = 0.025)
    sparsify : bool
        Sparsify the graph after the Kron reduction. (default is True)
    epsilon : float
        Sparsification parameter if the sparsification is used. (default = min(2/sqrt(G.N), 0.1))

    Returns
    -------
    Cs : ndarray

    """
    from pygsp.filters import Filter

    # TODO @ function
    if not epsilon:
        epsilon = min(10./sqrt(G.N), .1)

    Gs = [G]
    for i in range(Nlevels):
        L_reg = Gs[i].L.todense() + lamda*np.eye(Gs[i].N)
        _, Vtemp = np.linalg.eig(L_reg)
        V = np.ravel(Vtemp[:, 0])

        # Select the bigger group
        V = np.where(V >= 0, 1, 0)
        if np.sum(V) >= Gs[i].N/2.:
            ind = (V).nonzero()[0]
        else:
            ind = (1-V).nonzero()[0]

        if sparsify:
            Gtemp = kron_reduction(Gs[i], ind)
            Gs.append(utils.graph_sparsify(Gtemp, max(epsilon, 2./sqrt(Gs[i].N))))
        else:
            Gs.append(kron_reduction(Gs[i], ind))

        Gs[i+1].pyramid = {'ind': ind,
                           'green_kernel': Filter(Gs[i + 1], filters=[lambda x: 1./(lamda + x)]),
                           'level': i+1,
                           'K_reg': kron_reduction(L_reg, ind)}

    return Gs


def kron_reduction(G, ind):
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

    from pygsp.graphs import Graph

    if isinstance(G, Graph):
        if hasattr(G, 'lap_type'):
            if not G.lap_type == 'combinatorial':
                raise ValueError('Not implemented.')

        if G.directed:
            raise ValueError('This method only work for undirected graphs.')
        L = G.L.todense()

    else:
        L = G

    N = np.shape(L)[0]
    ind_comp = np.setdiff1d(np.arange(N), ind)

    L_red = L[np.ix_(ind, ind)]
    L_in_out = L[np.ix_(ind, ind_comp)]
    L_out_in = L[np.ix_(ind_comp, ind)]
    L_comp = L[np.ix_(ind_comp, ind_comp)]

    Lnew = L_red - np.dot(L_in_out, np.linalg.solve(L_comp, L_out_in))

    # Make the laplacian symetric if it is almost symetric!
    if np.sum(np.abs(Lnew-Lnew.T)) < np.spacing(1)*np.sum(np.abs(Lnew)):
        Lnew = (Lnew + Lnew.T)/2.

    if isinstance(G, Graph):
        # Suppress the diagonal ? This is a good question?
        Wnew = np.diag(np.diag(Lnew)) - Lnew
        Snew = np.diag(Lnew) - np.sum(Wnew, axis=0).T
        if np.linalg.norm(Snew, 2) < np.spacing(1000):
            Snew = 0
        Wnew = Wnew + np.diagonal(Wnew)
        Gnew = Graph(W=Wnew, coords=G.coords[ind, :],
                                  type='Kron reduction')
        G.copy_graph_attributes(ctype=False, Gn=Gnew)

    else:
        Gnew = Lnew

    return Gnew


def pyramid_analysis(Gs, f, filters=None, **kwargs):
    r"""
    Compute the graph pyramid transform coefficients

    Parameters
    ----------
    Gs : list of graph
        A multiresolution sequence of graph structures.
    f : ndarray
        Graph signal to analyze.
    kwargs : Dict
        Optional parameters that will be used
    filters : list
        A list of filter that will be used for the analysis and sythesis operator. If only one filter is given, it will be used for all levels. You may change that later on.

    Returns
    -------
    ca : ndarray
        Array with the coarse approximation at each level
    pe : ndarray
        Array with the prediction errors at each level
    """

    from pygsp.filters import Filter

    if np.shape(f)[0] != Gs[0].N:
        raise ValueError("The signal to analyze should have the same dimension as the first graph")

    Nlevels = len(Gs) - 1
    # check if the type of filters is right.
    if filters:
        if type(filters) != list:
            print('filters is not a list. I will convert it for you.')
            if hasattr(filters, '__call__'):
                filters = [filters]
            else:
                print('filters must be a list of function.')

        if len(filters) == 1:
            for _ in range(Nlevels-1):
                filters.append(filters[0])

        elif (1 < len(filters) and len(filters) < Nlevels) or Nlevels < len(filters):
            raise ValueError('The numbers of filters can must be one or equal to Nlevels')

    elif not filters:
        filters = []
        for i in range(Nlevels):
            filters.append(lambda x: .5/(.5+x))

    for i in range(Nlevels):
        Gs[i + 1].pyramid['filters'] = Filter(Gs[i + 1], filters=[filters[i]])

    # ca = [np.ravel(f)]
    ca = [f]
    pe = []

    for i in range(Nlevels):
        # Low pass the signal
        s_low = Gs[i+1].pyramid['filters'].analysis(Gs[i], ca[i], **kwargs)
        # Keep only the coefficient on the selected nodes
        ca.append(s_low[Gs[i+1].pyramid['ind']])
        # Compute prediction
        s_pred = interpolate(Gs[i], Gs[i+1], ca[i+1], **kwargs)
        # Compute errors
        pe.append(ca[i] - s_pred)

    try:
        pe.append(np.zeros((Gs[Nlevels].N, np.shape(f)[1])))
    except IndexError:
        pe.append(np.zeros((Gs[Nlevels].N)))

    return ca, pe


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid

    Parameters
    ----------
    ca : ndarray
        Array with the coarse approximation at each level
    pe : ndarray
        Array with the prediction errors at each level

    Returns
    -------
    coeff : ndarray
        Array of coefficient
    """
    Nl = len(ca) - 1
    N = 0

    for ele in ca:
        N += np.shape(ele)[0]

    try:
        Nt, Nv = np.shape(ca[Nl])
        coeff = coeff = np.zeros((N, Nv))
    except ValueError:
        Nt = np.shape(ca[Nl])[0]
        coeff = np.zeros((N))

    coeff[:Nt] = ca[Nl]

    ind = Nt
    for i in range(Nl):
        Nt = np.shape(ca[Nl - 1 - i])[0]
        coeff[ind+np.arange(Nt)] = pe[Nl - 1 - i]
        ind += Nt

    if ind != N:
        raise ValueError('Something is wrong here: contact the gspbox team.')

    return coeff


def pyramid_synthesis(Gs, coeff, order=100, **kwargs):
    r"""
    Synthesizes a signal from its graph pyramid transform coefficients

    Parameters
    ----------
    Gs : A multiresolution sequence of graph structures.
    coeff : ndarray
        The coefficients to perform the reconstruction
    order : int
        Degree of the Chebyshev approximation. (default = 100)

    Returns
    -------
    signal : The synthesized signal.
    ca : Cell array with the coarse approximation at each level
    """
    Nl = len(Gs) - 1

    # Initisalization
    Nt = Gs[Nl].N
    ca = [coeff[:Nt]]

    ind = Nt
    # Reconstruct each level
    for i in range(Nl):
        # Compute prediction
        Nt = Gs[Nl - 1 - i].N
        # Compute the ca coeff
        s_pred = interpolate(Gs[Nl - 1 - i], Gs[Nl - i], ca[i], order=order,
                             **kwargs)

        ca.append(s_pred + coeff[ind + np.arange(Nt)])
        ind += Nt

    ca.reverse()
    signal = ca[0]

    return signal, ca


def interpolate(Gh, Gl, coeff, order=100, **kwargs):
    r"""
    Interpolate lower coefficient

    Parameters
    ----------
    Gh : Upper graph
    Gl : Lower graph
    coeff : Coefficients
    order : int
        Degree of the Chebyshev approximation. (default = 100)

    Returns
    -------
    s_pred : Predicted signal

    """
    alpha = np.dot(Gl.pyramid['K_reg'], coeff)

    try:
        Nv = np.shape(coeff)[1]
        s_pred = np.zeros((Gh.N, Nv))
    except IndexError:
        s_pred = np.zeros((Gh.N))

    s_pred[Gl.pyramid['ind']] = alpha

    return Gl.pyramid['green_kernel'].analysis(Gh, s_pred, order=order,
                                               **kwargs)


def modulate(G, f, k):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal (column)
    k :  int
        Indices of frequencies

    Returns
    -------
    fm : Modulated signal

    """
    nt = np.shape(f)[1]
    fm = np.sqrt(G.N)*np.kron(np.ones((nt, 1)), f)*np.kron(np.ones((1, nt)), G.U[:, k])

    return fm


def translate(G, f, i):
    r"""
    Tranlate the signal f to the node i

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal
    i : int
        Indices of vertex

    Returns
    -------
    ft : translate signal

    """

    fhat = gft(G, f)
    nt = np.shape(f)[1]

    ft = np.sqrt(G.N)*igft(G, fhat, np.kron(np.ones((1, nt)), G.U[i]))

    return ft


def tree_multiresolution(G, Nlevel, reduction_method='resistance_distance',
                         compute_full_eigen=False, root=None):
    r"""
    Compute a multiresolution of trees

    Parameters
    ----------
    G : Graph
        Graph structure of a tree.
    Nlevel : Number of times to downsample and coarsen the tree
    root : int
        The index of the root of the tree. (default = 1)
    reduction_method : str
        The graph reduction method (default = 'resistance_distance')
    compute_full_eigen : bool
        To also compute the graph Laplacian eigenvalues for every tree in the sequence

    Returns
    -------
    Gs : ndarray
        Ndarray, with each element containing a graph structure represent a reduced tree.
    subsampled_vertex_indices : ndarray
        Indices of the vertices of the previous tree that are kept for the subsequent tree.

    """
    from pygsp.graphs import Graph

    if not root:
        if hasattr(G, 'root'):
            root = G.root
        else:
            root = 1

    Gs = [G]

    if compute_full_eigen:
        Gs[0] = compute_fourier_basis(G)

    subsampled_vertex_indices = []
    depths, parents = utils.tree_depths(G.A, root)
    old_W = G.W

    for lev in range(Nlevel):
        # Identify the vertices in the even depths of the current tree
        down_odd = round(depths) % 2
        down_even = np.ones((Gs[lev].N)) - down_odd
        keep_inds = np.where(down_even == 1)[0]
        subsampled_vertex_indices.append(keep_inds)

        # There will be one undirected edge in the new graph connecting each
        # non-root subsampled vertex to its new parent. Here, we find the new
        # indices of the new parents
        non_root_keep_inds, new_non_root_inds = np.setdiff1d(keep_inds, root)
        old_parents_of_non_root_keep_inds = parents[non_root_keep_inds]
        old_grandparents_of_non_root_keep_inds = parents[old_parents_of_non_root_keep_inds]
        # TODO new_non_root_parents = dsearchn(keep_inds, old_grandparents_of_non_root_keep_inds)

        old_W_i_inds, old_W_j_inds, old_W_weights = sparse.find(old_W)
        i_inds = np.concatenate((new_non_root_inds, new_non_root_parents))
        j_inds = np.concatenate((new_non_root_parents, new_non_root_inds))
        new_N = np.sum(down_even)

        if reduction_method == "unweighted":
            new_weights = np.ones(np.shape(i_inds))

        elif reduction_method == "sum":
            # TODO old_weights_to_parents_inds = dsearchn([old_W_i_inds,old_W_j_inds], [non_root_keep_inds, old_parents_of_non_root_keep_inds]);
            old_weights_to_parents = old_W_weights[old_weights_to_parents_inds]
            # old_W(non_root_keep_inds,old_parents_of_non_root_keep_inds);
            # TODO old_weights_parents_to_grandparents_inds = dsearchn([old_W_i_inds, old_W_j_inds], [old_parents_of_non_root_keep_inds, old_grandparents_of_non_root_keep_inds])
            old_weights_parents_to_grandparents = old_W_weights[old_weights_parents_to_grandparents_inds]
            # old_W(old_parents_of_non_root_keep_inds,old_grandparents_of_non_root_keep_inds);
            new_weights = old_weights_to_parents + old_weights_parents_to_grandparents
            new_weights = np.concatenate((new_weights. new_weights))

        elif reduction_method == "resistance_distance":
            # TODO old_weights_to_parents_inds = dsearchn([old_W_i_inds, old_W_j_inds], [non_root_keep_inds, old_parents_of_non_root_keep_inds])
            old_weights_to_parents = old_W_weight[sold_weights_to_parents_inds]
            # old_W(non_root_keep_inds,old_parents_of_non_root_keep_inds);
            # TODO old_weights_parents_to_grandparents_inds = dsearchn([old_W_i_inds, old_W_j_inds], [old_parents_of_non_root_keep_inds, old_grandparents_of_non_root_keep_inds])
            old_weights_parents_to_grandparents = old_W_weights[old_weights_parents_to_grandparents_inds]
            # old_W(old_parents_of_non_root_keep_inds,old_grandparents_of_non_root_keep_inds);
            new_weights = 1./(1./old_weights_to_parents + 1./old_weights_parents_to_grandparents)
            new_weights = np.concatenate(([new_weights, new_weights]))

        else:
            raise ValueError('Unknown graph reduction method.')

        new_W = sparse.csc_matrix((new_weights, (i_inds, j_inds)),
                                  shape=(new_N, new_N))
        # Update parents
        new_root = np.where(keep_inds == root)[0]
        parents = np.zeros(np.shape(keep_inds)[0], np.shape(keep_inds)[0])
        parents[:new_root - 1, new_root:] = new_non_root_parents

        # Update depths
        depths = depths[keep_inds]
        depths = depths/2.

        # Store new tree
        Gtemp = Graph(new_W, coords=Gs[lev].coords[keep_inds], limits=G.limits, gtype='tree',)
        Gtemp.L = create_laplacian(Gs[lev + 1], G.lap_type)
        Gtemp.root = new_root
        Gtemp = gsp_copy_graph_attributes(Gs[lev], False, Gs[lev + 1])

        if compute_full_eigen:
            Gs[lev + 1] = gsp_compute_fourier_basis(Gs[lev + 1])

        # Replace current adjacency matrix and root
        Gs.append(Gtemp)

        old_W = new_W
        root = new_root

    return Gs, subsampled_vertex_indices


def prox_tv(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, verbose=1, maxit=200, use_matrix=True):
    r"""
    TV proximal operator for graphs.

    This function computes the TV proximal operator for graphs. The TV norm
    is the one norm of the gradient. The gradient is defined in the
    function |gsp_grad|. This function require the PyUNLocBoX to be executed.

    Parameters
    ----------
    x: int
        Description.
    gamma: array_like
        Description.
    G: graph object
        Description.
    A: lambda function
        Description.
    At: lambda function
        Description.
    nu: float
        Description.
    tol: float
        Description.
    verbose: int
        Description.
    maxit: int
        Description.
    use_matrix: bool
        Description.

    Returns
    -------
    sol: solution
        Description.

    Examples
    --------
    TODO

    """

    if A is None:
        A = lambda x: x
    if At is None:
        At = lambda x: x

    if not hasattr(G, 'v_in'):
        adj2vec(G)

    tight = 0
    l1_nu = 2 * G.lmax * nu

    if use_matrix:
        l1_a = lambda x: G.Diff * A(x)
        l1_at = lambda x: G.Diff * At(D.T * x)
    else:
        l1_a = lambda x: grad(G, A(x))
        l1_at = lambda x: div(G, x)

    pyunlocbox.prox_l1(x, gamma, A=l1_a, At=l1_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
