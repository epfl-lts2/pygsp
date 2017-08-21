# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import transforms  # prevent circular import in Python < 3.5


logger = utils.build_logger(__name__)


def localize(g, i):
    r"""
    Localize a kernel g to the node i.

    Parameters
    ----------
    g : Filter
        kernel (or filterbank)
    i : int
        Index of vertex

    Returns
    -------
    gt : ndarray
        Translated signal

    """
    N = g.G.N
    f = np.zeros((N))
    f[i - 1] = 1

    gt = np.sqrt(N) * g.analysis(f)

    return gt


def modulate(G, f, k):
    r"""
    Modulation the signal f to the frequency k.

    Parameters
    ----------
    G : Graph
    f : ndarray
        Signal (column)
    k :  int
        Index of frequencies

    Returns
    -------
    fm : ndarray
        Modulated signal

    """
    nt = np.shape(f)[1]
    fm = np.sqrt(G.N) * np.kron(np.ones((nt, 1)), f) * \
        np.kron(np.ones((1, nt)), G.U[:, k])

    return fm


def translate(G, f, i):
    r"""
    Translate the signal f to the node i.

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

    raise NotImplementedError('Current implementation is not working.')

    fhat = transforms.gft(G, f)
    nt = np.shape(f)[1]

    ft = np.sqrt(G.N) * transforms.igft(G, fhat, np.kron(np.ones((1, nt)), G.U[i]))

    return ft
