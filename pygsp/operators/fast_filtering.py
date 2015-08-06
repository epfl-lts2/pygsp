# -*- coding: utf-8 -*-

from pygsp.graphs.gutils import estimate_lmax
from pygsp import utils

import numpy as np
from math import pi

logger = utils.build_logger(__name__)


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

    a1 = float(a_arange[1] - a_arange[0])/2
    a2 = float(a_arange[1] + a_arange[0])/2

    twf_old = signal
    twf_cur = (G.L.dot(signal) - a2 * signal)/a1

    for i in range(Nscales):
        r[np.arange(G.N) + G.N*i] = 0.5*c[i][0]*twf_old + c[i][1]*twf_cur
    for k in range(2, M + 1):
        twf_new = (2./a1) * (G.L.dot(twf_cur) - a2*twf_cur) - twf_old
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


def lanczos():
    raise NotImplementedError
