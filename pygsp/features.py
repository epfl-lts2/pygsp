# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.features` module implements different feature extraction
techniques based on :mod:`pygsp.graphs` and :mod:`pygsp.filters`.
"""

import numpy as np

from pygsp import filters, utils


def compute_avg_adj_deg(G):
    r"""
    Compute the average adjacency degree for each node.

    The average adjacency degree is the average of the degrees of a node and
    its neighbors.

    Parameters
    ----------
    G: Graph
        Graph on which the statistic is extracted
    """
    return np.sum(np.dot(G.A, G.A), axis=1) / (np.sum(G.A, axis=1) + 1.)


@utils.filterbank_handler
def compute_tig(g, **kwargs):
    r"""
    Compute the Tig for a given filter or filter bank.

    .. math:: T_ig(n) = g(L)_{i, n}

    Parameters
    ----------
    g: Filter
        One of :mod:`pygsp.filters`.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """
    return g.compute_frame()


@utils.filterbank_handler
def compute_norm_tig(g, **kwargs):
    r"""
    Compute the :math:`\ell_2` norm of the Tig.
    See :func:`compute_tig`.

    Parameters
    ----------
    g: Filter
        The filter or filter bank.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """
    tig = compute_tig(g, **kwargs)
    return np.linalg.norm(tig, axis=1, ord=2)


def compute_spectrogram(G, atom=None, M=100, **kwargs):
    r"""
    Compute the norm of the Tig for all nodes with a kernel shifted along the
    spectral axis.

    Parameters
    ----------
    G : Graph
        Graph on which to compute the spectrogram.
    atom : func
        Kernel to use in the spectrogram (default = exp(-M*(x/lmax)²)).
    M : int (optional)
        Number of samples on the spectral scale. (default = 100)
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.filter` method.
    """

    if not atom:
        def atom(x):
            return np.exp(-M * (x / G.lmax)**2)

    scale = np.linspace(0, G.lmax, M)
    spectr = np.empty((G.N, M))

    for shift_idx in range(M):
        shift_filter = filters.Filter(G, lambda x: atom(x - scale[shift_idx]))
        tig = compute_norm_tig(shift_filter, **kwargs).squeeze()**2
        spectr[:, shift_idx] = tig

    G.spectr = spectr
    return spectr
