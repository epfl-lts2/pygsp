# -*- coding: utf-8 -*-
r"""This module contains functionalities for the reduction of graphs' vertex set while keeping the graph structure."""

from pygsp.utils import resistance_distance, build_logger
from pygsp.graphs import Graph
from pygsp.filters import Filter

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import eigs, spsolve

logger = build_logger(__name__)


def graph_sparsify(M, epsilon, maxiter=10):
    r"""
    Sparsify a graph using Spielman-Srivastava algorithm.

    Parameters
    ----------
    M : Graph or sparse matrix
        Graph structure or a Laplacian matrix
    epsilon : int
        Sparsification parameter

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    Note
    ----
    Epsilon should be between 1/sqrt(N) and 1

    Examples
    --------
    >>> from pygsp import graphs, operators
    >>> G = graphs.Sensor(256, Nc=20, distribute=True)
    >>> epsilon = 0.4
    >>> G2 = operators.graph_sparsify(G, epsilon)

    Reference
    ---------
    See :cite: `spielman2011graph` `rudelson1999random` `rudelson2007sampling`
    for more informations

    """
    # Test the input parameters
    if isinstance(M, Graph):
        if not M.lap_type == 'combinatorial':
            raise NotImplementedError
        L = M.L
    else:
        L = M

    N = np.shape(L)[0]

    if not 1./np.sqrt(N) <= epsilon < 1:
        raise ValueError('GRAPH_SPARSIFY: Epsilon out of required range')

    # pas sparse
    resistance_distances = resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(M, Graph):
        W = M.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sparse.csc_matrix(W)

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    # Calculate the new weights.
    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    for i in range(maxiter):
        # Rudelson, 1996 Random Vectors in the Isotropic Position
        # (too hard to figure out actual C0)
        C0 = 1 / 30.
        # Rudelson and Vershynin, 2007, Thm. 3.1
        C = 4 * C0
        q = round(N * np.log(N) * 9 * C**2 / (epsilon**2))

        results = stats.rv_discrete(values=(np.arange(np.shape(Pe)[0]), Pe)).rvs(size=q)
        spin_counts = stats.itemfreq(results).astype(int)
        per_spin_weights = weights / (q * Pe)

        counts = np.zeros(np.shape(weights)[0])
        counts[spin_counts[:, 0]] = spin_counts[:, 1]
        new_weights = counts * per_spin_weights

        sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                     shape=(N, N))
        sparserW = sparserW + sparserW.T
        sparserL = sparse.diags(sparserW.diagonal(), 0) - sparserW

        if Graph(W=sparserW).is_connected():
            break
        elif i == maxiter - 1:
            logger.warning('Despite attempts to reduce epsilon, sparsified graph is disconnected')
        else:
            epsilon -= (epsilon - 1/np.sqrt(N)) / 2.

    if isinstance(M, Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not M.directed:
            sparserW = (sparserW + sparserW.T) / 2.

        Mnew = Graph(W=sparserW)
        M.copy_graph_attributes(Mnew)
    else:
        Mnew = sparse.lil_matrix(sparserL)

    return Mnew


def interpolate(G, f_subsampled, keep_inds, order=100, reg_eps=0.005, **kwargs):
    r"""
    Interpolation of a graph signal.

    Parameters
    ----------
    G : Graph
    f_subsampled : ndarray
        A graph signal on the graph G.
    keep_inds : ndarray
        List of indices on which the signal is sampled.
    order : int
        Degree of the Chebyshev approximation (default = 100).
    reg_eps : float
        The regularized graph Laplacian is $\bar{L}=L+\epsilon I$.
        A smaller epsilon may lead to better regularization,
        but will also require a higher order Chebyshev approximation.

    Returns
    -------
    f_interpolated : ndarray
        Interpolated graph signal on the full vertex set of G.

    References: pesenson2009variational

    """
    L_reg = G.L + reg_eps * sparse.eye(G.N)
    K_reg = getattr(G.mr, 'K_reg', kron_reduction(L_reg, keep_inds))
    green_kernel = getattr(G.mr, 'green_kernel',
                           Filter(G, filters=lambda x: 1. / (reg_eps + x)))

    alpha = K_reg.dot(f_subsampled)

    try:
        Nv = np.shape(f_subsampled)[1]
        f_interpolated = np.zeros((G.N, Nv))
    except IndexError:
        f_interpolated = np.zeros((G.N))

    f_interpolated[keep_inds] = alpha

    return green_kernel.analysis(f_interpolated, order=order, **kwargs)


def graph_multiresolution(G, levels, **kwargs):
    r"""
    Compute a pyramid of graphs using the kron reduction.

    'graph_multiresolution(G,levels)' computes a multiresolution of
    graph by repeatedly downsampling and performing graph reduction. The
    default downsampling method is the largest eigenvector method based on
    the polarity of the components of the eigenvector associated with the
    largest graph Laplacian eigenvalue. The default graph reduction method
    is Kron reduction followed by a graph sparsification step.
    *param* is a structure of optional parameters.

    Parameters
    ----------
    G : Graph structure
        The graph to reduce.
    levels : int
        Number of level of decomposition
    params : dict
        lambd : float
            Stability parameter. It adds self loop to the graph to give the
            algorithm some stability (default = 0.025). [UNUSED?!]
        sparsify : bool
            To perform a spectral sparsification step immediately after
            the graph reduction (default is True).
        sparsify_eps : float
            Parameter epsilon used in the spectral sparsification
            (default is min(10/sqrt(G.N),.3)).
        downsampling_method: string
            The graph downsampling method (default is 'largest_eigenvector').
        reduction_method : string
            The graph reduction method (default is 'kron')
        compute_full_eigen : bool
            To also compute the graph Laplacian eigenvalues and eigenvectors
            for every graph in the multiresolution sequence (default is False).
        reg_eps : float
            The regularized graph Laplacian is $\bar{L}=L+\epsilon I$.
            A smaller epsilon may lead to better regularization, but will also
            require a higher order Chebyshev approximation. (default is 0.005)

    Returns
    -------
    Gs : ndarray
        The graph layers.

    Example
    -------
    >>> import pygsp
    >>> levels = 5
    >>> G = pygsp.graphs.Sensor(N=256)
    >>> Gs = pygsp.operators.graph_multiresolution(G, levels)
    >>> for idx in range(levels):
    ...     Gs[i].plotting['plot_name'] = 'Reduction level: {}'.format(idx)
    ...     Gs[i].plot()

    """
    # lambd = float(kwargs.pop('lambd', 0.025))
    sparsify = bool(kwargs.pop('sparsify', True))
    sparsify_eps = float(kwargs.pop('sparsify_eps', min(10./np.sqrt(G.N), 0.3)))
    downsampling_method = kwargs.pop('downsampling_method', 'largest_eigenvector')
    reduction_method = kwargs.pop('downsampling_method', 'kron')
    compute_full_eigen = bool(kwargs.pop('compute_full_eigen', False))
    reg_eps = float(kwargs.pop('reg_eps', 0.005))

    if compute_full_eigen:
        if not hasattr(G, 'e') or not hasattr(G, 'U'):
            G.compute_fourier_basis()
    else:
        if not hasattr(G, 'lmax'):
            G.estimate_lmax()

    Gs = [G]
    Gs[0].mr = {'idx': np.arange(G.N), 'orig_idx': np.arange(G.N)}

    for i in range(levels):
        if downsampling_method == 'largest_eigenvector':
            if hasattr(Gs[i], 'U'):
                V = Gs[i].U[:, -1]
            else:
                V = eigs(Gs[i].L, 1)[1][:, 0]

            V *= np.sign(V[0])
            ind = np.nonzero(V >= 0)[0]

        else:
            raise NotImplementedError('Unknown graph downsampling method.')

        if reduction_method == 'kron':
            Gs.append(kron_reduction(Gs[i], ind))

        else:
            raise NotImplementedError('Unknown graph reduction method.')

        if sparsify and Gs[i+1].N > 2:
            Gs[i+1] = graph_sparsify(Gs[i+1], min(max(sparsify_eps, 2. / np.sqrt(Gs[i+1].N)), 1.))
            # TODO : Make in place modifications instead!

        if compute_full_eigen:
            Gs[i+1].compute_fourier_basis()
        else:
            Gs[i+1].estimate_lmax()

        Gs[i+1].mr = {'idx': ind, 'orig_idx': Gs[i].mr['orig_idx'][ind], 'level': i}

        L_reg = Gs[i].L + reg_eps * sparse.eye(Gs[i].N)
        Gs[i].mr['K_reg'] = kron_reduction(L_reg, ind)
        Gs[i].mr['green_kernel'] = Filter(Gs[i], filters=lambda x: 1./(reg_eps + x))

    return Gs


def kron_reduction(G, ind):
    r"""
    Compute the kron reduction.

    This function perform the Kron reduction of the weight matrix in the
    graph *G*, with boundary nodes labeled by *ind*. This function will
    create a new graph with a weight matrix Wnew that contain only boundary
    nodes and is computed as the Schur complement of the original matrix
    with respect to the selected indices.

    Parameters
    ----------
    G : Graph or sparse matrix
        Graph structure or weight matrix
    ind : list
        indices of the nodes to keep

    Returns
    -------
    Gnew : Graph or sparse matrix
        New graph structure or weight matrix


    References: dorfler2013kron

    """
    if isinstance(G, Graph):
        if hasattr(G, 'lap_type'):
            if not G.lap_type == 'combinatorial':
                message = 'Unknwon reduction for {} laplacian.'.format(G.lap_type)
                raise NotImplementedError(message)

        if G.directed:
            message = 'This method only work for undirected graphs.'
            raise NotImplementedError(message)

        if not hasattr(G, 'L'):
            G.compute_laplacian()
        L = G.L

    else:
        L = G

    N = np.shape(L)[0]
    ind_comp = np.setdiff1d(np.arange(N, dtype=int), ind)

    L_red = L[np.ix_(ind, ind)]
    L_in_out = L[np.ix_(ind, ind_comp)]
    L_out_in = L[np.ix_(ind_comp, ind)].tocsc()
    L_comp = L[np.ix_(ind_comp, ind_comp)].tocsc()

    Lnew = L_red - L_in_out.dot(spsolve(L_comp, L_out_in))

    # Make the laplacian symmetric if it is almost symmetric!
    if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
        Lnew = (Lnew + Lnew.T) / 2.

    if isinstance(G, Graph):
        # Suppress the diagonal ? This is a good question?
        Wnew = sparse.diags(Lnew.diagonal(), 0) - Lnew
        Snew = Lnew.diagonal() - np.ravel(Wnew.sum(0))
        if np.linalg.norm(Snew, 2) >= np.spacing(1000):
            Wnew = Wnew + sparse.diags(Snew, 0)

        coords = G.coords[ind, :] if len(G.coords.shape) else np.ndarray(None)
        Gnew = Graph(W=Wnew, coords=coords, lap_type=G.lap_type,
                     plotting=G.plotting, gtype='Kron reduction')
    else:
        Gnew = Lnew

    return Gnew


def pyramid_analysis(Gs, f, **kwargs):
    r"""
    Compute the graph pyramid transform coefficients.

    Parameters
    ----------
    Gs : list of graphs
        A multiresolution sequence of graph structures.
    f : ndarray
        Graph signal to analyze.
    h_filters : list
        A list of filter that will be used for the analysis and sythesis operator.
        If only one filter is given, it will be used for all levels.
        Default is h(x) = 1 / (2x+1)

    Returns
    -------
    ca : ndarray
        Coarse approximation at each level
    pe : ndarray
        Prediction error at each level
    h_filters : list
        Graph spectral filters applied

    References: shuman2013framework pesenson2009variational

    """
    if np.shape(f)[0] != Gs[0].N:
        raise ValueError("PYRAMID ANALYSIS: The signal to analyze should have the same dimension as the first graph.")

    levels = len(Gs) - 1

    # check if the type of filters is right.
    h_filters = kwargs.pop('h_filters', lambda x: 1. / (2*x+1))

    if not isinstance(h_filters, list):
        if hasattr(h_filters, '__call__'):
            logger.warning('Converting filters into a list.')
            h_filters = [h_filters]
        else:
            logger.error('Filters must be a list of functions.')

    if len(h_filters) == 1:
        h_filters = h_filters * levels

    elif len(h_filters) != levels:
        message = 'The number of filters must be one or equal to {}.'.format(levels)
        raise ValueError(message)

    ca = [f]
    pe = []

    for i in range(levels):
        # Low pass the signal
        s_low = Filter(Gs[i], filters=h_filters[i]).analysis(ca[i], **kwargs)
        # Keep only the coefficient on the selected nodes
        ca.append(s_low[Gs[i+1].mr['idx']])
        # Compute prediction
        s_pred = interpolate(Gs[i], ca[i+1], Gs[i+1].mr['idx'], **kwargs)
        # Compute errors
        pe.append(ca[i] - s_pred)

    return ca, pe


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid.
    NOT NECESSARY ANYMORE.

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
        coeff = np.zeros((N, Nv))
    except ValueError:
        Nt = np.shape(ca[Nl])[0]
        coeff = np.zeros((N))

    coeff[:Nt] = ca[Nl]
    ind = Nt

    for i in range(Nl):
        Nt = np.shape(ca[Nl - 1 - i])[0]
        tmpNt = np.arange(Nt, dtype=int)
        coeff[ind + tmpNt] = pe[Nl - 1 - i]
        ind += Nt

    if ind != N:
        raise ValueError('Something is wrong here: contact the gspbox team.')

    return coeff


def pyramid_synthesis(Gs, cap, pe, order=30, **kwargs):
    r"""
    Synthesize a signal from its graph pyramid transform coefficients.

    Parameters
    ----------
    Gs : Array of Graphs
        A multiresolution sequence of graph structures.
    cap : ndarray
        Coarsest approximation of the original signal.
    pe : ndarray
        Prediction error at each level.
    use_exact : bool
        To use exact graph spectral filtering instead of the Chebyshev approximation.
    order : int
        Degree of the Chebyshev approximation (default=30).
    least_squares : bool
        To use the least squares synthesis (default=False).
    h_filters : ndarray
        The filters used in the analysis operator.
        These are required for least squares synthesis, but not for the direct synthesis method.
    use_landweber : bool
        To use the Landweber iteration approximation in the least squares synthesis.
    reg_eps : float
        Interpolation parameter.
    landweber_its : int
        Number of iterations in the Landweber approximation for least squares synthesis.
    landweber_tau : float
        Parameter for the Landweber iteration.

    Returns
    -------
    reconstruction : ndarray
        The reconstructed signal.
    ca : ndarray
        Coarse approximations at each level

    """
    least_squares = bool(kwargs.pop('least_squares', False))
    def_ul = Gs[0].N > 3000 or not hasattr(Gs[0], 'e') or not hasattr(Gs[0], 'U')
    use_landweber = bool(kwargs.pop('use_landweber', def_ul))
    reg_eps = float(kwargs.get('reg_eps', 0.005))

    if least_squares and 'h_filters' not in kwargs:
        ValueError('h-filters not provided.')

    levels = len(Gs) - 1
    if len(pe) != levels:
        ValueError('Gs and pe have different shapes.')

    ca = [cap]

    # Reconstruct each level
    for i in range(levels):

        if not least_squares:
            s_pred = interpolate(Gs[levels - i - 1], ca[i], Gs[levels - i].mr['idx'],
                                 order=order, reg_eps=reg_eps, **kwargs)
            ca.append(s_pred + pe[levels - i - 1])

        else:
            ca.append(pyramid_single_interpolation(Gs[levels - i - 1], ca[i],
                      pe[levels - i - 1], h_filters[levels - i - 1],
                      use_landweber=use_landweber, **kwargs))

    ca.reverse()
    reconstruction = ca[0]

    return reconstruction, ca


def pyramid_single_interpolation(G, ca, pe, keep_inds, h_filter, **kwargs):
    r"""
    Sythesize a single level of the graph pyramid transform.

    Parameters
    ----------
    G : Graph
        Graph structure on which the signal resides.
    ca : ndarray
        Coarse approximation of the signal on a reduced graph.
    pe : ndarray
        Prediction error that was made when forming the current coarse approximation.
    keep_inds : ndarray
        The indices of the vertices to keep when downsampling the graph and signal.
    h_filter : lambda expression
        The filter in use at this level.
    use_landweber : bool
        To use the Landweber iteration approximation in the least squares synthesis.
        Default is False.
    reg_eps : float
        Interpolation parameter. Default is 0.005.
    landweber_its : int
        Number of iterations in the Landweber approximation for least squares synthesis.
        Default is 50.
    landweber_tau : float
        Parameter for the Landweber iteration. Default is 1.

    Returns
    -------
    finer_approx : 
        Coarse approximation of the signal on a higher resolution graph.

    """
    nb_ind = keep_inds.shape
    N = G.N
    reg_eps = float(kwargs.pop('reg_eps', 0.005))
    use_landweber = bool(kwargs.pop('use_landweber', False))
    landweber_its = int(kwargs.pop('landweber_its', 50))
    landweber_tau = float(kwargs.pop('landweber_tau', 1.))

    # index matrix (nb_ind x N) of keep_inds, S_i,j = 1 iff keep_inds[i] = j
    S = sparse.csr_matrix(([1] * nb_ind, (range(nb_ind), keep_inds)), shape=(nb_ind, N))

    if use_landweber:
        x = np.zeros(N)
        z = np.concatenate((ca, pe), axis=0)
        green_kernel = Filter(G, filters=lambda x: 1./(x+reg_eps))
        PhiVlt = green_kernel.analysis(S.T, **kwargs).T
        filt = Filter(G, filters=h_filter, **kwargs)

        for iteration in range(landweber_its):
            h_filtered_sig = filt.analysis(x, **kwargs)
            x_bar = h_filtered_sig[keep_inds]
            y_bar = x - interpolate(G, x_bar, keep_inds, **kwargs)
            z_delt = np.concatenate((x_bar, y_bar), axis=0)
            z_delt = z - z_delt
            alpha_new = PhiVlt * z_delt[nb_ind:]
            x_up = sparse.csr_matrix((z_delt, (range(nb_ind), [1] * nb_ind)), shape=(N, 1))
            reg_L = G.L + reg_esp * sparse.eye(N)

            elim_inds = np.setdiff1d(np.arange(N, dtype=int), keep_inds)
            L_red = reg_L[np.ix_(keep_inds, keep_inds)]
            L_in_out = reg_L[np.ix_(keep_inds, elim_inds)]
            L_out_in = reg_L[np.ix_(elim_inds, keep_inds)]
            L_comp = reg_L[np.ix_(elim_inds, elim_inds)]

            next_term = L_red * alpha_new - L_in_out * spsolve(L_comp, L_out_in * alpha_new)
            next_up = sparse.csr_matrix((next_term, (keep_inds, [1] * nb_ind)), shape=(N, 1))
            x += landweber_tau * filt.analysis(x_up - next_up, **kwargs) + z_delt[nb_ind:]

        finer_approx = x

    else:
        # When the graph is small enough, we can do a full eigendecomposition
        # and compute the full analysis operator T_a
        if not hasattr(G, 'e') or not hasattr(G, 'U'):
            G.compute_fourier_basis()
        H = G.U * sparse.diags(h_filter(G.e), 0) * G.U.T
        Phi = G.U * sparse.diags(1./(reg_eps + G.e), 0) * G.U.T
        Ta = np.concatenate((S * H, sparse.eye(G.N) - Phi[:, keep_inds] * spsolve(Phi[np.ix_(keep_inds, keep_inds)], S*H)), axis=0)
        finer_approx = spsolve(Ta.T * Ta, Ta.T * np.concatenate((ca, pe), axis=0))


def tree_depths(A, root):
    r"""Empty docstring. TODO."""
    if not Graph(A=A).is_connected():
        raise ValueError('Graph is not connected')

    N = np.shape(A)[0]
    assigned = root - 1
    depths = np.zeros((N))
    parents = np.zeros((N))

    next_to_expand = np.array([root])
    current_depth = 1

    while len(assigned) < N:
        new_entries_whole_round = []
        for i in range(len(next_to_expand)):
            neighbors = np.where(A[next_to_expand[i]])[0]
            new_entries = np.setdiff1d(neighbors, assigned)
            parents[new_entries] = next_to_expand[i]
            depths[new_entries] = current_depth
            assigned = np.concatenate((assigned, new_entries))
            new_entries_whole_round = np.concatenate((new_entries_whole_round,
                                                      new_entries))

        current_depth = current_depth + 1
        next_to_expand = new_entries_whole_round

    return depths, parents


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
        Gs[0].compute_fourier_basis()

    subsampled_vertex_indices = []
    depths, parents = tree_depths(G.A, root)
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
        Gtemp = Graph(new_W, coords=Gs[lev].coords[keep_inds], limits=G.limits, gtype='tree', root=new_root)
        Gs[lev].copy_graph_attributes(Gtemp, False)

        if compute_full_eigen:
            Gs[lev + 1].compute_fourier_basis()

        # Replace current adjacency matrix and root
        Gs.append(Gtemp)

        old_W = new_W
        root = new_root

    return Gs, subsampled_vertex_indices
