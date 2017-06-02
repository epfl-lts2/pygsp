# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from skimage.util import view_as_windows, pad
from pyflann import *
from .. import Graph
from ... import utils


class ImgPatches(Graph):
    r"""
    Create a nearest neighbors graph from patches of an image.

    Parameters
    ----------
    img : array
        Input image.
    patch_shape : tuple, optional
        Dimensions of the patch window. Syntax: (height, width), or (height,),
        in which case width = height.
    n_nbrs : int, optional
        Number of neighbors to consider
    dist_type : string, optional
        Type of distance between patches to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.

    Examples
    --------
    >>> from pygsp.graphs import nngraphs
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::2, ::2])
    >>> G = nngraphs.ImgPatches(img)

    """

    def __init__(self, img, patch_shape=(3, 3), n_nbrs=8,
                 dist_type='euclidean', **kwargs):
        try:
            h, w, d = img.shape
        except ValueError:
            try:
                h, w = img.shape
                d = 1
            except ValueError:
                print("Image should be a 2-d array.")

        try:
            r, c = patch_shape
        except ValueError:
            r = patch_shape[0]
            c = r
        if d <= 1:
            pad_width = ((int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                         (int((c - 0.5) / 2.), int((c + 0.5) / 2.)))
        else:
            pad_width = ((int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                         (int((c - 0.5) / 2.), int((c + 0.5) / 2.)),
                         (0, 0))
        img_pad = pad(img, pad_width=pad_width, mode='symmetric')

        patches = view_as_windows(img_pad,
                                  window_shape=tuple(np.maximum((r, c, d), 1)))
        X = patches.reshape((h * w, r * c * d))

        set_distance_type(dist_type)
        flann = FLANN()
        nbrs, dists = flann.nn(X, X, num_neighbors=(n_nbrs + 1),
                               algorithm="kmeans", branching=32, iterations=7,
                               checks=16)

        node_list = [[i] * n_nbrs for i in range(h * w)]
        node_list = [item for sublist in node_list for item in sublist]
        nbrs = nbrs[:, 1:].reshape((len(node_list),))
        dists = dists[:, 1:].reshape((len(node_list),))

        # This line guarantees that the median weight is 0.5:
        weights = np.exp(np.log(0.5) * dists / (np.median(dists)))

        W = sparse.csc_matrix((weights, (node_list, nbrs)),
                              shape=(h * w, h * w))
        W = utils.symmetrize(W, 'full')

        super(ImgPatches, self).__init__(W=W, gtype='patch-graph',
                                         perform_all_checks=False, **kwargs)

        self.img = img
