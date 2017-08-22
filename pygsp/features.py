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
def compute_tig(filt, **kwargs):
    r"""
    Compute the Tig for a given filter or filterbank.

    .. math:: T_ig(n) = g(L)_{i, n}

    Parameters
    ----------
    filt: Filter object
        The filter or filterbank.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.analysis` method.
    """
    signals = np.eye(filt.G.N)
    return filt.analysis(signals, **kwargs)


@utils.filterbank_handler
def compute_norm_tig(filt, **kwargs):
    r"""
    Compute the :math:`\ell_2` norm of the Tig.
    See :func:`compute_tig`.

    Parameters
    ----------
    filt: Filter
        The filter or filterbank.
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.analysis` method.
    """
    tig = compute_tig(filt, **kwargs)
    return np.linalg.norm(tig, axis=1, ord=2)


def compute_spectrogram(G, atom=None, M=100, **kwargs):
    r"""
    Compute the norm of the Tig for all nodes with a kernel shifted along the
    spectral axis.

    Parameters
    ----------
    G : Graph
        Graph on which to compute the spectrogram.
    atom : Filter kernel (optional)
        Kernel to use in the spectrogram (default = exp(-M*(x/lmax)²)).
    M : int (optional)
        Number of samples on the spectral scale. (default = 100)
    kwargs: dict
        Additional parameters to be passed to the
        :func:`pygsp.filters.Filter.analysis` method.
    """

    if not atom:
        def atom(x):
            return np.exp(-M * (x / G.lmax)**2)

    scale = np.linspace(0, G.lmax, M)
    spectr = np.empty((G.N, M))

    for shift_idx in range(M):
        shift_filter = filters.Filter(G, lambda x: atom(x - scale[shift_idx]))
        spectr[:, shift_idx] = compute_norm_tig(shift_filter, **kwargs)**2

    G.spectr = spectr
    return spectr
