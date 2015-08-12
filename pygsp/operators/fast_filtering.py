# -*- coding: utf-8 -*-

from pygsp.graphs.gutils import estimate_lmax
from pygsp import utils

import numpy as np
import scipy as sp
from math import pi

logger = utils.build_logger(__name__)


@utils.filterbank_handler
def compute_cheby_coeff(f, G=None, m=30, N=None, i=0, *args):
    r"""
    Compute Chebyshev coefficients for a Filterbank

    Parameters
    ---------
    f : List of filters
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
        G.lmax = estimate_lmax(G)
        logger.info('The variable lmax has not been computed yet, it will be done.)')

    a_arange = [0, G.lmax]

    a1 = (a_arange[1] - a_arange[0])/2
    a2 = (a_arange[1] + a_arange[0])/2
    c = np.zeros((m + 1))

    for o in range(m + 1):
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
        logger.error("The signal has an invalid shape")

    if not hasattr(G, 'lmax'):
        G.lmax = estimate_lmax(G)

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


def lanczos_op(fi, s, G=None, order=30):
    r"""
    Perform the lanczos approximation of the signal s

    Parameters
    ----------
    fi: Filter
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

    try:
        Nv = np.shape(s)[1]
        is2d = True
        c = np.zeros((G.N*Nf, Nv))
    except IndexError:
        Nv = 1
        is2d = False
        c = np.zeros((G.N*Nf))

    for j in range(Nv):
        if is2d:
            V, H, _ = lanczos(G.L.toarray(), order, s[:, j])
        else:
            V, H, _ = lanczos(G.L.toarray(), order, s)

        Eh, Uh = np.linalg.eig(H)

        # Eh = np.diagonal(Eh)
        Eh = np.where(Eh < 0, 0, Eh)
        fie = fi.evaluate(Eh)
        V = np.dot(V, Uh)

        for i in range(Nf):
            if is2d:
                c[np.arange(G.N) + i*G.N, j] = np.dot(V, fie[:][i] * np.dot(V.T, s[:, j]))
            else:
                c[np.arange(G.N) + i*G.N] = np.dot(V, fie[:][i] * np.dot(V.T, s))

    return c


def lanczos(A, order, x):
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
        r -= np.dot(V, np.dot(V.T, r)) # full reorthogonalization
        H[k + 1, k + hiv] = np.linalg.norm(r, axis=0)
        orth[k] = np.linalg.norm(np.dot(V.T, V) - M)

    H = H[np.ix_(np.arange(order), np.arange(order))]

    return V, H, orth


def _sum_ind(ind1, ind2):
    ind = np.tile(np.ravel(ind1), (np.size(ind2), 1)).T + np.ravel(ind2)
    return np.ravel(ind)
