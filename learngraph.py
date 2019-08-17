import numpy as np
from scipy import sparse
from scipy.spatial.distance import squareform
from time import time

def isvector(x):
    '''Test if x is a vector'''
    if sparse.issparse(x):
        return x.shape[1]==1
    try:
        return len(x.shape)==1
    except:
        return False


    
def prox_sum_log(x, gamma):
    '''Solves the proximal problem:
        sol = argmin_z 0.5*||x - z||_2^2 - gamma * sum(log(z))
    The solution is given by
        sol = (x + sqrt(x.^2 + 4*gamma)) /2
    '''
    sol = (x + np.sqrt(x*x + 4*gamma)) /2
    return sol


def issymetric(W, tol=1e-7):
    '''Test if a sparse matrix is symmetric'''
    WT = sparse.triu(W) - sparse.triu(W.transpose())
    return np.sum(np.abs(WT))<tol

def squareform_sp(w, safe=True):
    '''Squareform function for sparse matrix'''
    if not(sparse.issparse(w)):
        return squareform(w)
    if isvector(w):
        ## VECTOR -> MATRIX
        l = w.shape[0]
        n = np.int(np.round((1 + np.sqrt(1+8*l))/2))
        # check input
        if not(l == n*(n-1)//2):
            raise ValueError('Bad vector size!')

        ind_vec, _, data = sparse.find(w)

        num_nz = len(ind_vec)

        # indices inside the matrix
        ind_i = np.zeros((num_nz))
        ind_j = np.zeros((num_nz))

        curr_row = 0
        offset = 0
        # length of current row of matrix, counting from after the diagonal
        sl = n - 1
        for ii in range(num_nz):
            ind_vec_i = ind_vec[ii]
            # if we change row, the vector index is bigger by at least the
            # length of the line + the offset.
            while(ind_vec_i >= (sl + offset)):
                offset = offset + sl
                sl = sl - 1
                curr_row = curr_row + 1
        
            ind_i[ii] = curr_row
            ind_j[ii] = ind_vec_i - offset + (n-sl)
        indx = np.concatenate((ind_i, ind_j))
        indy = np.concatenate((ind_j, ind_i))
        data = np.concatenate((data, data))
        w = sparse.csr_matrix((data, (indx, indy)), (n, n))
        return w
    else:
        ## MATRIX -> VECTOR
        # first checks
        assert(len(w.shape)==2)
        m, n = w.shape
        if not(m == n) or np.sum(np.abs(w.diagonal()))>0:
            raise ValueError('Matrix has to be square with zero diagonal!');
        if safe:
            assert(issymetric(w))
        ind_i, ind_j, s = sparse.find(w);
        # keep only upper triangular part
        ind_upper = ind_i < ind_j
        ind_i = ind_i[ind_upper]+1
        ind_j = ind_j[ind_upper]+1
        s = s[ind_upper]
        # compute new (vector) index from [i,j] (matrix) indices
        new_ind = ind_j + (ind_i-1)*n - ind_i*(ind_i+1)//2-1
        w = sparse.csr_matrix((s,(new_ind, np.zeros(new_ind.shape))), shape=(n*(n-1)//2, 1))
    return w


def sum_squareform(n, mask=None):
    '''sparse matrix that sums the squareform of a vector
    
    Input parameters:
          n:    size of matrix W
          mask: if given, S only contain the columns indicated by the mask
    
    Output parameters:
          S:    matrix so that S*w = sum(W) for vector w = squareform(W)
          St:   the adjoint of S
    
    Creates sparse matrices S, St = S' so that
        S*w = sum(W),       where w = squareform(W)
    
    The mask is used for large scale computations where only a few
    non-zeros in W are to be summed. It needs to be the same size as w,
    n(n-1)/2 elements.
    
    Properties of S:
    * size(S) = [n, (n(n-1)/2)]     % if no mask is given.
    * size(S, 2) = nnz(w)           % if mask is given
    * norm(S)^2 = 2(n-1)
    * sum(S) = 2*ones(1, n*(n-1)/2)
    * sum(St) = sum(squareform(mask))   -- for full mask = (n-1)*ones(n,1)
    '''
    if mask is not None:
        if not(mask.shape[0] == n*(n-1)//2):
            raise ValueError('mask size has to be n(n-1)/2');

        ind_vec, _ = mask.nonzero()

        # final number of columns is the nnz(mask)
        ncols = len(ind_vec)

        # indices inside the matrix
        I = np.zeros((ncols)).astype(np.int)
        J = np.zeros((ncols)).astype(np.int)
        
        curr_row = 0
        offset = 0
        # length of current row of matrix, counting from after the diagonal
        sl = n - 1;
        for ii in range(ncols):
            ind_vec_i = ind_vec[ii]
            # if we change row, the vector index is bigger by at least the
            # length of the line + the offset.
            while(ind_vec_i >= (sl + offset)):
                offset = offset + sl
                sl = sl - 1
                curr_row = curr_row + 1
            I[ii] = curr_row
            J[ii] = ind_vec_i - offset + (n-sl)
    else:
        # number of columns is the length of w given size of W
        ncols = ((n-1)*n)//2;

        I = np.zeros((ncols)).astype(np.int)
        J = np.zeros((ncols)).astype(np.int)

        # offset
        k = 0;
        for i in range(1,n):
            I[k: k + (n-i)] = np.arange(i, n).astype(np.int)
            k = k + (n-i)
        k = 0;
        for i in range(1,n):
            J[k: k + (n-i)] = i-1
            k = k + (n-i)
    indx = np.concatenate((I,J))
    indy = np.concatenate((np.arange(ncols),np.arange(ncols)))
    data = np.ones((2*ncols))
    S = sparse.csr_matrix((data, (indx, indy)), shape=(n,ncols), dtype=np.int)
    St = sparse.csr_matrix((data, (indy, indx)), shape=(ncols, n), dtype=np.int)
    return S, St


def lin_map(X, lims_out, lims_in=None):
    '''Map linearly from a given range to another.'''
    
    if lims_in is None:
        lims_in = [np.min(X), np.max(X)];

    a = lims_in[0];
    b = lims_in[1];
    c = lims_out[0];
    d = lims_out[1];
    
    Y = (X-a) * ((d-c)/(b-a)) + c;
    return Y


def norm_S(S):
    '''Norm of the matrix given by the function sum_squareform'''
    return np.sqrt(2*np.max(np.sum(S, axis=1)))


def compute_theta_bounds(Z, geom_mean=False, islist=None):
    '''Compute the values of parameter theta (controlling sparsity) 
    that should be expected to give each sparsity level. Return upper 
    and lower bounds each sparsity level k=[1, ..., n-1] neighbors/node.
    
    Z : distance matrix
    geom_mean: use geometric mean instead of arithmetic mean? default: False
    '''
    # Z is the zero-diagonal pairwise distance matrix between nodes
    
    if isvector(Z):
        ValueError('Z must be a matrix.')
    assert(len(Z.shape)==2)
    n = len(Z);
    
    if islist is None:
        islist = not(Z.shape[0]==Z.shape[1])
    if islist:
        Z_sorted = np.sort(Z, axis=1)
    else:
        assert(Z.shape[0]==Z.shape[1])
        # don't take into account the diagonal of Z
        Z_sorted = np.zeros((n, n-1));
        for i in range(n):
            Z_sorted[i, :] = np.sort(np.concatenate((Z[i,:i],Z[i,i+1:])), kind='mergesort')

    n, m  = Z_sorted.shape
    
    B_k = np.cumsum(Z_sorted, axis=1)       # cummulative sum for each row
    K_mat = np.tile(np.arange(1,m+1), (n, 1))
    ## Theoretical intervals of theta for each desired sparsity level:
    if geom_mean:
        theta_u = np.mean(1./(np.sqrt(K_mat*Z_sorted*Z_sorted - B_k*Z_sorted)+1e-15), axis=0)
    else:
        # try geometric mean instead of arithmetic:
        theta_u = np.exp(np.mean(np.log(1/(np.sqrt(K_mat*Z_sorted*Z_sorted - B_k*Z_sorted)+1e-15)), axis=0))
    theta_l = np.zeros(theta_u.shape)
    theta_l[:-1] = theta_u[1:]
    return theta_l, theta_u, Z_sorted

def gsp_compute_graph_learning_theta(Z, k, geom_mean=False, islist=None):

    
    # Z is the zero-diagonal pairwise distance matrix between nodes
    theta_min, theta_max, _ = compute_theta_bounds(Z, geom_mean, islist)
    theta_min = theta_min[k-1];
    theta_max = theta_max[k-1];

    if k > 1:
        theta = np.sqrt(theta_min * theta_max);
    else:
        theta = theta_min * 1.1;
    return theta, theta_min, theta_max


def learn_graph_log_degree(Z,
    a = 1,
    b = 1,
    c = 0,
    verbosity = 1,
    maxit = 1000,
    tol = 1e-5,
    step_size = .5, 
    fix_zeros = False,
    max_w = np.inf,
    edge_mask = None,
    w_0 = 0,
    rel_edge=1e-5):
    
    if isvector(Z):
        z = Z;
    else:
        z = squareform_sp(Z)

    l = z.shape[0] # number of edges
    # n(n-1)/2 = l => n = (1 + sqrt(1+8*l))/ 2
    n = int(np.round((1 + np.sqrt(1+8*l))/ 2))  # number of nodes

    if not(w_0==0):
        if c==0:
            raise ValueError('When w_0 is specified, c should not be 0');
        if not isvector(w_0):
            w_0 = squareform_sp(w_0)
    else:
        w_0 = 0;

    # if sparsity pattern is fixed we optimize with respect to a smaller number
    # of variables, all included in w
    if edge_mask is not None:
        if not(isvector(edge_mask)):
            edge_mask = squareform_sp(edge_mask)
        # use only the non-zero elements to optimize
        ind, _ = edge_mask.nonzero()

        z = z[ind].data
        if not(np.isscalar(w_0)):
            w_0 = w_0[ind].data
    
    w = np.zeros(z.shape);

    ## Needed operators
    # S*w = sum(W)

    S, St = sum_squareform(n, mask=edge_mask)

    # S: edges -> nodes
    K_op = lambda w : S.dot(w)

    # S': nodes -> edges
    Kt_op = lambda z: St.dot(z)

    norm_K = norm_S(S)


    ## Learn the graph
    # min_{W>=0}     tr(X'*L*X) - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2, where L = diag(sum(W))-W
    # min_W       I{W>=0} + W(:)'*Dx(:)  - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2
    # min_W                f(W)          +       g(L_op(W))      +   h(W)

    # put proximal of trace plus positivity together
    feval = lambda w: 2*np.sum(w*z) # half should be counted
    fprox = lambda w, gamma: np.minimum(max_w, np.maximum(0, w - 2*gamma*z)) # weighted soft thresholding op

    geval = lambda v: -a * np.sum(np.log(v+1e-15))
    gprox = lambda v, gamma: prox_sum_log(v, gamma*a)
    # proximal of conjugate of g: v-gamma*gprox(v/gamma, 1/gamma)
    g_star_prox = lambda v, gamma: v - gamma*a * prox_sum_log(v/(gamma*a), 1/(gamma*a))

    if w_0 == 0:
        # "if" not needed, for c = 0 both are the same but gains speed
        heval = lambda w: b * np.sum(w*w)
        hgrad = lambda w: 2 * b * w
        hbeta = 2 * b;
    else:
        heval = lambda w: b * np.sum(w*w) + c * np.sum((w - w_0)**2)
        hgrad = lambda w: 2 * ((b+c) * w - c * w_0)
        hbeta = 2 * (b+c)


    ## Custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
    # parameters mu, epsilon for convergence (see [1])
    mu = hbeta + norm_K;     #TODO: is it squared or not??
    epsilon = lin_map(0.0, [0, 1/(1+mu)], [0,1]);   # in (0, 1/(1+mu) )

    # INITIALIZATION
    # primal variable ALREADY INITIALIZED
    # dual variable
    v_n = K_op(w)
        
    stat = dict()
    stat['time'] = np.nan
    if verbosity > 1:
        stat['f_eval'] = np.empty((maxit))
        stat['g_eval'] = np.empty((maxit))
        stat['h_eval'] = np.empty((maxit))
        stat['fgh_eval'] = np.empty((maxit))
        stat['pos_violation'] = np.empty((maxit))

    if verbosity > 1:
        print('Relative change of primal, dual variables, and objective fun\n');

    tstart = time()
    gn = lin_map(step_size, [epsilon, (1-epsilon)/mu], [0,1]) # in [epsilon, (1-epsilon)/mu]
    for i in range(maxit):
        Y_n = w - gn * (hgrad(w) + Kt_op(v_n))
        y_n = v_n + gn * (K_op(w))
        P_n = fprox(Y_n, gn)
        p_n = g_star_prox(y_n, gn) # = y_n - gn*g_prox(y_n/gn, 1/gn)
        Q_n = P_n - gn * (hgrad(P_n) + Kt_op(p_n))
        q_n = p_n + gn * (K_op(P_n))

        if verbosity > 1:
            stat['f_eval'][i] = feval(w)
            stat['g_eval'][i] = geval(K_op(w))
            stat['h_eval'][i] = heval(w)
            stat['fgh_eval'][i] = stat['f_eval'][i] + stat['g_eval'][i] + stat['h_eval'][i]
            stat['pos_violation'][i] = -np.sum(np.minimum(0,w))

        rel_norm_primal = np.linalg.norm(- Y_n + Q_n)/(np.linalg.norm(w)+1e-15)
        rel_norm_dual = np.linalg.norm(- y_n + q_n)/(np.linalg.norm(v_n)+1e-15)

        if verbosity > 3:
            print('iter {:4d}: {:6.4e}   {:6.4e}   {:6.3e}'.format(i, rel_norm_primal, rel_norm_dual, stat['fgh_eval'][i]))
        elif verbosity > 2:
            print('iter {:4d}: {%6.4e}   {:6.4e}   {:6.3e}'.format(i, rel_norm_primal, rel_norm_dual, stat['fgh_eval'][i]))
        elif verbosity > 1:
            print('iter {:4d}: {:6.4e}   {:6.4e}'.format(i, rel_norm_primal, rel_norm_dual))

        w = w - Y_n + Q_n
        v_n = v_n - y_n + q_n

        if (rel_norm_primal < tol) and (rel_norm_dual < tol):
            break

    stat['time'] = time()-tstart
    if verbosity > 0:
        print('# iters: {:4d}. Rel primal: {:6.4e} Rel dual: {:6.4e}  OBJ {:6.3e}'.format(
                i, rel_norm_primal, rel_norm_dual, feval(w) + geval(K_op(w)) + heval(w)))
        print('Time needed is {:f} seconds'.format(stat['time']))
    
    # Force positivity on the solution
    
    w[w<rel_edge*np.max(w)]=0
    
    if edge_mask is not None:
        indw = w>0
        w = sparse.csr_matrix((w[indw], (ind[indw], np.zeros((np.sum(indw))))),  (l, 1))

    if isvector(Z):
        W = w;
    else:
        W = squareform_sp(w);

    return W, stat
