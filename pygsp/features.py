# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.features` module implements different feature extraction
techniques based on :mod:`pygsp.graphs` and :mod:`pygsp.filters`.
"""

import numpy as np

from .graphs import Graph
from .filters import Filter
from .utils import filterbank_handler


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
    if not isinstance(G, Graph):
        raise ValueError("Graph object expected as first argument.")

    return np.sum(np.dot(G.A, G.A), axis=1) / (np.sum(G.A, axis=1) + 1.)


@filterbank_handler
def compute_tig(filt, method=None, **kwargs):
    r"""
    Compute the Tig for a given filter or filterbank.

    .. math:: T_ig(n) = g(L)_{i, n}

    Parameters
    ----------
    filt: Filter object
        The filter (or filterbank) to localize
    method: string (optional)
        Which method to use. Accept 'cheby', 'exact'.
        Default : 'exact' if filt.G has U and e defined, otherwise 'cheby'
    i: int (optional)
        Index of the filter to analyse (default: 0)
    """
    if not isinstance(filt, Filter):
        raise ValueError("Filter object expected as first argument.")

    signals = np.eye(filt.G.N)
    return filt.analysis(signals, method=method, **kwargs)


@filterbank_handler
def compute_norm_tig(filt, method=None, *args, **kwargs):
    r"""
    Compute the :math:`\ell_2` norm of the Tig.
    See :func:`compute_tig`.

    Parameters
    ----------
    filt: Filter
        The filter (or filterbank)
    method: string (optional)
        Which method to use. Accept 'cheby', 'exact'
        (default : 'exact' if filt.G has U and e defined, otherwise 'cheby')
    """
    tig = compute_tig(filt, method=method, *args, **kwargs)
    return np.linalg.norm(tig, axis=1, ord=2)


def compute_spectrogram(G, atom=None, M=100, method=None, **kwargs):
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

    """
    if not atom or not hasattr(atom, '__call__'):
        def atom(x):
            return np.exp(-M * (x / G.lmax)**2)

    scale = np.linspace(0, G.lmax, M)
    spectr = np.zeros((G.N, M))

    for shift_idx in range(M):
        shft_filter = Filter(G,
                             filters=[lambda x: atom(x - scale[shift_idx])],
                             **kwargs)
        spectr[:, shift_idx] = compute_norm_tig(shft_filter, method=method)**2

    G.spectr = spectr
    return spectr
