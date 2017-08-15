# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.features` module implements different feature extraction
techniques based on :mod:`pygsp.graphs` and :mod:`pygsp.filters`.
"""

import numpy as np

from .graphs import Graph
from .filters import Filter
from .utils import filterbank_handler
from skimage.util import view_as_windows, pad


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
    from pygsp.filters import Filter

    if not hasattr(G, 'lmax'):
        G.estimate_lmax()

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


def patch_features(img, patch_shape=(3, 3)):
    r"""
    Compute a patch feature vector for every pixel of an image.

    Parameters
    ----------
    img : array
        Input image.
    patch_shape : tuple, optional
        Dimensions of the patch window. Syntax: (height, width), or (height,),
        in which case width = height.

    Returns
    -------
    array
        Feature matrix.

    Notes
    -----
    The feature vector of a pixel `i` will consist of the stacking of the
    intensity values of all pixels in the patch centered at `i`, for all color
    channels. So, if the input image has `d` color channels, the dimension of
    the feature vector of each pixel is (patch_shape[0] * patch_shape[1] * d).

    Examples
    --------
    >>> from pygsp import features
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::2, ::2])
    >>> X = features.patch_features(img)

    """

    try:
        h, w, d = img.shape
    except ValueError:
        try:
            h, w = img.shape
            d = 0
        except ValueError:
            print("Image should be at least a 2-d array.")

    try:
        r, c = patch_shape
    except ValueError:
        r = patch_shape[0]
        c = r
    if d == 0:
        pad_width = ((int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                     (int((c - 0.5) / 2.), int((c + 0.5) / 2.)))
        window_shape = (r, c)
        d = 1  # For the reshape in the return call
    else:
        pad_width = ((int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                     (int((c - 0.5) / 2.), int((c + 0.5) / 2.)),
                     (0, 0))
        window_shape = (r, c, d)
    # Pad the image
    img_pad = pad(img, pad_width=pad_width, mode='symmetric')

    # Extract patches
    patches = view_as_windows(img_pad, window_shape=window_shape)

    return patches.reshape((h * w, r * c * d))
