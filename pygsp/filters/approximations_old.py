# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils


_logger = utils.build_logger(__name__)


@utils.filterbank_handler
def compute_cheby_coeff(f, m=30, N=None, *args, **kwargs):
    r"""
    Compute Chebyshev coefficients for a Filterbank.

    Parameters
    ----------
    f : Filter
        Filterbank with at least 1 filter
    m : int
        Maximum order of Chebyshev coeff to compute
        (default = 30)
    N : int
        Grid order used to compute quadrature
        (default = m + 1)
    i : int
        Index of the Filterbank element to compute
        (default = 0)

    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients

    """
    G = f.G
    i = kwargs.pop('i', 0)

    if not N:
        N = m + 1

    a_arange = [0, G.lmax]

    a1 = (a_arange[1] - a_arange[0]) / 2
    a2 = (a_arange[1] + a_arange[0]) / 2
    c = np.zeros(m + 1)

    tmpN = np.arange(N)
    num = np.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * np.dot(f._kernels[i](a1 * num + a2),
                               np.cos(np.pi * o * (tmpN + 0.5) / N))

    return c


def cheby_op(G, c, signal, **kwargs):
    r"""
    Chebyshev polynomial of graph Laplacian applied to vector.

    Parameters
    ----------
    G : Graph
    c : ndarray or list of ndarrays
        Chebyshev coefficients for a Filter or a Filterbank
    signal : ndarray
        Signal to filter

    Returns
    -------
    r : ndarray
        Result of the filtering

    """
    # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    c = np.atleast_2d(c)
    Nscales, M = c.shape

    if M < 2:
        raise TypeError("The coefficients have an invalid shape")

    # thanks to that, we can also have 1d signal.
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N * Nscales, Nv))
    except IndexError:
        r = np.zeros((G.N * Nscales))

    a_arange = [0, G.lmax]

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.

    twf_old = signal
    twf_cur = (G.L.dot(signal) - a2 * signal) / a1

    tmpN = np.arange(G.N, dtype=int)
    for i in range(Nscales):
        r[tmpN + G.N*i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

    factor = 2/a1 * (G.L - a2 * sparse.eye(G.N))
    for k in range(2, M):
        twf_new = factor.dot(twf_cur) - twf_old
        for i in range(Nscales):
            r[tmpN + G.N*i] += c[i, k] * twf_new

        twf_old = twf_cur
        twf_cur = twf_new

    return r


def cheby_rect(G, bounds, signal, **kwargs):
    r"""
    Fast filtering using Chebyshev polynomial for a perfect rectangle filter.

    Parameters
    ----------
    G : Graph
    bounds : array-like
        The bounds of the pass-band filter
    signal : array-like
        Signal to filter
    order : int (optional)
        Order of the Chebyshev polynomial (default: 30)

    Returns
    -------
    r : array-like
        Result of the filtering

    """
    if not (isinstance(bounds, (list, np.ndarray)) and len(bounds) == 2):
        raise ValueError('Bounds of wrong shape.')

    bounds = np.array(bounds)

    m = int(kwargs.pop('order', 30) + 1)

    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N, Nv))
    except IndexError:
        r = np.zeros((G.N))

    b1, b2 = np.arccos(2. * bounds / G.lmax - 1.)
    factor = 4./G.lmax * G.L - 2.*sparse.eye(G.N)

    T_old = signal
    T_cur = factor.dot(signal) / 2.
    r = (b1 - b2)/np.pi * signal + 2./np.pi * (np.sin(b1) - np.sin(b2)) * T_cur

    for k in range(2, m):
        T_new = factor.dot(T_cur) - T_old
        r += 2./(k*np.pi) * (np.sin(k*b1) - np.sin(k*b2)) * T_new
        T_old = T_cur
        T_cur = T_new

    return r


def compute_jackson_cheby_coeff(filter_bounds, delta_lambda, m):
    r"""
    To compute the m+1 coefficients of the polynomial approximation of an ideal band-pass between a and b, between a range of values defined by lambda_min and lambda_max.

    Parameters
    ----------
    filter_bounds : list
        [a, b]
    delta_lambda : list
        [lambda_min, lambda_max]
    m : int

    Returns
    -------
    ch : ndarray
    jch : ndarray

    References
    ----------
    :cite:`tremblay2016compressive`

    """
    # Parameters check
    if delta_lambda[0] > filter_bounds[0] or delta_lambda[1] < filter_bounds[1]:
        _logger.error("Bounds of the filter are out of the lambda values")
        raise()
    elif delta_lambda[0] > delta_lambda[1]:
        _logger.error("lambda_min is greater than lambda_max")
        raise()

    # Scaling and translating to standard cheby interval
    a1 = (delta_lambda[1]-delta_lambda[0])/2
    a2 = (delta_lambda[1]+delta_lambda[0])/2

    # Scaling bounds of the band pass according to lrange
    filter_bounds[0] = (filter_bounds[0]-a2)/a1
    filter_bounds[1] = (filter_bounds[1]-a2)/a1

    # First compute cheby coeffs
    ch = np.arange(float(m+1))
    ch[0] = (2/(np.pi))*(np.arccos(filter_bounds[0])-np.arccos(filter_bounds[1]))
    for i in ch[1:]:
        ch[i] = (2/(np.pi * i)) * \
            (np.sin(i * np.arccos(filter_bounds[0])) - np.sin(i * np.arccos(filter_bounds[1])))

    # Then compute jackson coeffs
    jch = np.arange(float(m+1))
    alpha = (np.pi/(m+2))
    for i in jch:
        jch[i] = (1/np.sin(alpha)) * \
            ((1 - i/(m+2)) * np.sin(alpha) * np.cos(i * alpha) +
             (1/(m+2)) * np.cos(alpha) * np.sin(i * alpha))

    # Combine jackson and cheby coeffs
    jch = ch * jch

    return ch, jch


def lanczos_op(f, s, order=30):
    r"""
    Perform the lanczos approximation of the signal s.

    Parameters
    ----------
    f: Filter
    s : ndarray
        Signal to approximate.
    order : int
        Degree of the lanczos approximation. (default = 30)

    Returns
    -------
    L : ndarray
        lanczos approximation of s

    """
    G = f.G
    Nf = len(f.g)

    # To have the right shape for the output array depending on the signal dim
    try:
        Nv = np.shape(s)[1]
        is2d = True
        c = np.zeros((G.N*Nf, Nv))
    except IndexError:
        Nv = 1
        is2d = False
        c = np.zeros((G.N*Nf))

    tmpN = np.arange(G.N, dtype=int)
    for j in range(Nv):
        if is2d:
            V, H, _ = lanczos(G.L.toarray(), order, s[:, j])
        else:
            V, H, _ = lanczos(G.L.toarray(), order, s)

        Eh, Uh = np.linalg.eig(H)

        Eh[Eh < 0] = 0
        fe = f.evaluate(Eh)
        V = np.dot(V, Uh)

        for i in range(Nf):
            if is2d:
                c[tmpN + i*G.N, j] = np.dot(V, fe[:][i] * np.dot(V.T, s[:, j]))
            else:
                c[tmpN + i*G.N] = np.dot(V, fe[:][i] * np.dot(V.T, s))

    return c


def lanczos(A, order, x):
    r"""
    TODO short description

    Parameters
    ----------
    A: ndarray

    Returns
    -------
    """
    try:
        N, M = np.shape(x)
    except ValueError:
        N = np.shape(x)[0]
        M = 1
        x = x[:, np.newaxis]

    # normalization
    q = np.divide(x, np.kron(np.ones((N, 1)), np.linalg.norm(x, axis=0)))

    # initialization
    hiv = np.arange(0, order*M, order)
    V = np.zeros((N, M*order))
    V[:, hiv] = q

    H = np.zeros((order + 1, M*order))
    r = np.dot(A, q)
    H[0, hiv] = np.sum(q*r, axis=0)
    r -= np.kron(np.ones((N, 1)), H[0, hiv])*q
    H[1, hiv] = np.linalg.norm(r, axis=0)

    orth = np.zeros((order))
    orth[0] = np.linalg.norm(np.dot(V.T, V) - M)

    for k in range(1, order):
        if np.sum(np.abs(H[k, hiv + k - 1])) <= np.spacing(1):
            H = H[:k - 1, _sum_ind(np.arange(k), hiv)]
            V = V[:, _sum_ind(np.arange(k), hiv)]
            orth = orth[:k]

            return V, H, orth

        H[k - 1, hiv + k] = H[k, hiv + k - 1]
        v = q
        q = r/np.tile(H[k - 1, k + hiv], (N, 1))
        V[:, k + hiv] = q

        r = np.dot(A, q)
        r -= np.tile(H[k - 1, k + hiv], (N, 1))*v
        H[k, k + hiv] = np.sum(np.multiply(q, r), axis=0)
        r -= np.tile(H[k, k + hiv], (N, 1))*q

        # The next line has to be checked
        r -= np.dot(V, np.dot(V.T, r))  # full reorthogonalization
        H[k + 1, k + hiv] = np.linalg.norm(r, axis=0)
        orth[k] = np.linalg.norm(np.dot(V.T, V) - M)

    H = H[np.ix_(np.arange(order), np.arange(order))]

    return V, H, orth


def _sum_ind(ind1, ind2):
    ind = np.tile(np.ravel(ind1), (np.size(ind2), 1)).T + np.ravel(ind2)
    return np.ravel(ind)
