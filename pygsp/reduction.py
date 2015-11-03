# -*- coding: utf-8 -*-

from pygsp.utils import resistance_distance, build_logger
from pygsp.graphs import Graph, gutils
from pygsp.filters import Filter

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import eigs, spsolve
from math import sqrt

logger = build_logger(__name__)


def graph_sparsify(M, epsilon):
    r"""
    Sparsify a graph using Spielman-Srivastava algorithm.

    Parameters
    ----------
    M : Graph or sparse matrix
        Graph structure or sparse matrix
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
    >>> from pygsp import graphs, reduction
    >>> G = graphs.Sensor(256, Nc=20, distribute=True)
    >>> epsilon = 0.4
    >>> G2 = reduction.graph_sparsify(G, epsilon)

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

    if not 1./sqrt(N) <= epsilon < 1:
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

    # Rudelson, 1996 Random Vectors in the Isotropic Position
    # (too hard to figure out actual C0)
    C0 = 1 / 30.
    # Rudelson and Vershynin, 2007, Thm. 3.1
    C = 4 * C0
    q = round(N * log(N) * 9 * C**2 / (epsilon**2))

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

    if isinstance(M, Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not M.directed:
            sparserW = (sparserW + sparserW.T) / 2.
            sparserL = (sparserL + sparserL.T) / 2.

        Mnew = Graph(W=sparserW, L=sparserL)
        M.copy_graph_attributes(Mnew)
    else:
        Mnew = sparse.lil_matrix(sparserL)

    return Mnew


def interpolate(Gu, Gl, coeff, order=100, **kwargs):
    r"""
    Interpolate lower coefficients.

    Parameters
    ----------
    Gu : Graph
        Upper graph
    Gl : Graph
        Lower graph
    coeff : list
        Coefficients
    order : int
        Degree of the Chebyshev approximation. (default = 100)

    Returns
    -------
    s_pred : Predicted signal

    """
    alpha = np.dot(Gl.pyramid['K_reg'].toarray(), coeff)

    try:
        Nv = np.shape(coeff)[1]
        s_pred = np.zeros((Gu.N, Nv))
    except IndexError:
        s_pred = np.zeros((Gu.N))

    s_pred[Gl.pyramid['ind']] = alpha

    return Gl.pyramid['green_kernel'].analysis(Gu, s_pred, order=order,
                                               **kwargs)


def kron_pyramid(G, Nlevels, **kwargs):
    r"""
    Compute a pyramid of graphs using the kron reduction.

    Parameters
    ----------
    G : Graph structure
    Nlevels : int
        Number of level of decomposition
    params : dict
        lambd : float
            Stability parameter. It add self loop to the graph to give the algorithm some stability (default = 0.025).
        sparsify : bool
            Sparsify the graph after the Kron reduction (default is True).
        epsilon : float
            Sparsification parameter if the sparsification is used (default = min(2/sqrt(G.N), 0.1)).
        filters : list
            A cell array of filter that will be used for the analysis and sythesis operators.
            If only one filter is given, it will be used for all levels. (default = 0.5 / (0.5 + x))

    Returns
    -------
    Cs : ndarray

    """
    lambd = float(kwargs.pop('lambd', 0.025))
    sparsify = bool(kwargs.pop('sparsify', True))
    epsilon = float(kwargs.pop('epsilon', min(10. / sqrt(G.N), .1)))
    filters = kwargs.pop('filters', lambda x: 1 / (2*x + 1.))

    if hasattr(filters, '__call__'):
        filters = [filters] * Nlevels

    if isinstance(filters, list):
        if len(filters) != Nlevels:
            raise ValueError('KRON_PYRAMID: Incorrect number of filters. Expected {}, got {}'.format(Nlevels, len(filters)))
    else:
        raise TypeError('KRON_PYRAMID: Filters expected to be a function or a list of functions.')

    Gs = [G]
    for i in range(Nlevels):
        L_reg = Gs[i].L + lambd * sparse.eye(Gs[i].N)
        V = eigs(L_reg, 1)[1][:, 0]

        # Select the biggest group
        if sum(np.sign(V)) >= 0:
            ind = np.nonzero(V >= 0)[0]
        else:
            ind = np.nonzero(V < 0)[0]

        if sparsify:
            Gtemp = kron_reduction(Gs[i], ind)
            Gs.append(graph_sparsify(Gtemp, max(epsilon, 2. / sqrt(Gs[i].N))))
        else:
            Gs.append(kron_reduction(Gs[i], ind))

        Gs[i + 1].pyramid = {'ind': ind,
                             'green_kernel': Filter(Gs[i + 1],
                                                    filters=[lambda x: 1./(lambd + x)]),
                             'filter': filters[i],
                             'level': i,
                             'K_reg': kron_reduction(L_reg, ind)}

    return Gs


def kron_reduction(G, ind):
    r"""
    Compute the kron reduction.

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

    """
    if isinstance(G, Graph):
        if hasattr(G, 'lap_type'):
            if not G.lap_type == 'combinatorial':
                raise ValueError('Not implemented.')

        if G.directed:
            raise ValueError('This method only work for undirected graphs.')
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

        Gnew = Graph(W=Wnew, coords=G.coords[ind, :],
                     type='Kron reduction')
        G.copy_graph_attributes(Gnew, ctype=False)

    else:
        Gnew = Lnew

    return Gnew


def pyramid_analysis(Gs, f, filters=None, **kwargs):
    r"""
    Compute the graph pyramid transform coefficients.

    Parameters
    ----------
    Gs : list of graph
        A multiresolution sequence of graph structures.
    f : ndarray
        Graph signal to analyze.
    kwargs : dict
        Optional parameters that will be used
    filters : list
        A list of filter that will be used for the analysis and sythesis operator.
        If only one filter is given, it will be used for all levels. You may change that later on.

    Returns
    -------
    ca : ndarray
        Array with the coarse approximation at each level
    pe : ndarray
        Array with the prediction errors at each level

    """
    if np.shape(f)[0] != Gs[0].N:
        raise ValueError("PYRAMID ANALYSIS: The signal to analyze should have the same dimension as the first graph.")

    Nlevels = len(Gs) - 1

    # check if the type of filters is right.
    if filters:
        if not isinstance(filters, list):
            if hasattr(filters, '__call__'):
                logger.warning('Converting filters into a list.')
                filters = [filters]
            else:
                logger.error('Filters must be a list of functions.')

        if len(filters) == 1:
            filters = filters * Nlevels

        elif len(filters) != Nlevels:
            raise ValueError('The number of filters must be one or equal to Nlevels.')
    else:
        filters = map(lambda idx: Gs[idx + 1].pyramid['filters'], range(Nlevels))

    # ca = [np.ravel(f)]
    ca = [f]
    pe = []

    for i in range(Nlevels):
        # Low pass the signal
        s_low = Filter(Gs[i], filters=[filters[i]]).analysis(Gs[i], ca[i], **kwargs)
        # Keep only the coefficient on the selected nodes
        ca.append(s_low[Gs[i].pyramid['ind'], :])
        # Compute prediction
        s_pred = interpolate(Gs[i], Gs[i + 1], ca[i + 1], **kwargs)
        # Compute errors
        pe.append(ca[i] - s_pred)

    try:
        pe.append(np.zeros((Gs[Nlevels].N, np.shape(f)[1])))
    except IndexError:
        pe.append(np.zeros((Gs[Nlevels].N)))

    return ca, pe


def pyramid_cell2coeff(ca, pe):
    r"""
    Cell array to vector transform for the pyramid.

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
    tmpNt = np.arange(Nt, dtype=int)
    for i in range(Nl):
        Nt = np.shape(ca[Nl - 1 - i])[0]
        coeff[ind + tmpNt] = pe[Nl - 1 - i]
        ind += Nt

    if ind != N:
        raise ValueError('Something is wrong here: contact the gspbox team.')

    return coeff


def pyramid_synthesis(Gs, coeff, order=100, **kwargs):
    r"""
    Synthesizes a signal from its graph pyramid transform coefficients.

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
        s_pred = interpolate(Gs[Nl - 1 - i], Gs[Nl - i], ca[Nl - i],
                             order=order, **kwargs)

        ca.append(s_pred + coeff[ind + np.arange(Nt)])
        ind += Nt

    ca.reverse()
    signal = ca[0]

    return signal, ca


def tree_depths(A, root):
    r"""
    Empty docstring. TODO
    """

    if gutils.check_connectivity(A) == 0:
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
            neighbors = np.where(A[next_to_expand[i]] > 1e-7)[0]
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
        Gs[0] = gutils.compute_fourier_basis(G)

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
            gutils.compute_fourier_basis(Gs[lev + 1])

        # Replace current adjacency matrix and root
        Gs.append(Gtemp)

        old_W = new_W
        root = new_root

    return Gs, subsampled_vertex_indices
