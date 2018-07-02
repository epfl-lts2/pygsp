"""Drafty stationarity module."""

import numpy as np
from scipy.interpolate import interp1d

from . import filters


def estimate_graph_psd(G, sig, Nrand=10, Npoint=30):
    """Estimate the power spectral density on graph.

    Parameters
    ----------
    G : graph
    sig : ndarray
        Signal whose PSD is to be estimated.
    Nrand : int
        Number of random signals used for the estimation.
    Npoint : int
        Number of points at which the PSD is estimated.

    """
    # Define filterbank.
    g = filters.Itersine(G, Nf=Npoint, overlap=2)
    mu = np.linspace(0, G.lmax, Npoint)

    # Filter signal.
    sig_filt = g.filter(sig, method='chebyshev', order=2 * Npoint)
    sig_dist = np.sum(sig_filt**2, axis=0)
    if sig_dist.ndim > 1:
        sig_dist = np.mean(sig_dist, axis=0).squeeze()

    # Estimate the eigenvectors by filtering random signals.
    rand_sig = np.random.binomial(n=1, p=0.5, size=[G.N, Nrand]) * 2 - 1
    rand_sig_filered = g.filter(
        rand_sig, method='chebyshev', order=2 * Npoint)
    eig_dist = np.mean(np.sum(rand_sig_filered**2, axis=0), axis=0).squeeze()

    # Compute PSD.
    psd_values = sig_dist / eig_dist
    inter = interp1d(mu, psd_values, kind='linear')

    return filters.Filter(G, inter), (mu, psd_values)
