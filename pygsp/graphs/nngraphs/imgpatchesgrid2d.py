# -*- coding: utf-8 -*-

from . import ImgPatches
import networkx as nx
import numpy as np


class ImgPatchesGrid2d(ImgPatches):
    r"""
    Create the union of an image patch graph with a 2-dimensional grid graph.

    Parameters
    ----------
    img : array
        Input image.
    patch_shape : tuple, optional
        Dimensions of the patch window. Syntax : (height, width).
    n_nbrs : int
        Number of neighbors to consider
    dist_type : string
        Type of distance between patches to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
    aggregation: callable, optional
        Function used for aggregating the weights of the patch graph and the
        2-d grid graph. Default is sum().

    Examples
    --------
    >>> from pygsp.graphs import nngraphs
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::2, ::2])
    >>> G = nngraphs.ImgPatchesGrid2d(img)

    """

    def __init__(self, img, patch_shape, n_nbrs,
                 aggregation=lambda Wp, Wg: Wp + Wg, **kwargs):
        super(ImgPatchesGrid2d, self).__init__(img=img,
                                               patch_shape=patch_shape,
                                               n_nbrs=n_nbrs,
                                               **kwargs)
        m, n = self.img.shape
        # Grid2d from pygsp is too slow:
        # from .. import Grid2d
        # Gg = Grid2d(Nv=n, Mv=m)
        # Wg = G.Wg
        # Use networkx instead:
        Gg = nx.grid_2d_graph(m, n)
        Wg = nx.to_scipy_sparse_matrix(Gg)  # some edges seem to be missing

        self.W = aggregation(self.W, Wg)

        x = np.kron(np.ones((m, 1)), (np.arange(n) / float(n)).reshape(n, 1))
        y = np.kron(np.ones((n, 1)), np.arange(m) / float(m)).reshape(m * n, 1)
        y = np.sort(y, axis=0)[::-1]
        self.coords = np.concatenate((x, y), axis=1)
        self.gtype = self.gtype + '-2d-grid'
