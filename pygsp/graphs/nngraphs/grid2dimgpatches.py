# -*- coding: utf-8 -*-

# prevent circular import in Python < 3.5
from pygsp.graphs import Graph, Grid2d, ImgPatches


class Grid2dImgPatches(Graph):
    r"""Union of a patch graph with a 2D grid graph.

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
    aggregate: callable, optional
        Function used for aggregating the weights Wp of the patch graph and the
        weigths Wg 2d grid graph. Default is :func:`lambda Wp, Wg: Wp + Wg`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.Grid2dImgPatches(img, use_flann=False)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, img, patch_shape=(3, 3), n_nbrs=8,
                 aggregate=lambda Wp, Wg: Wp + Wg, **kwargs):
        Gg = Grid2d(img.shape[0], img.shape[1], **kwargs)
        Gp = ImgPatches(img, patch_shape=patch_shape, n_nbrs=n_nbrs, **kwargs)
        gtype = '{}_{}'.format(Gg.gtype, Gp.gtype)
        super(Grid2dImgPatches, self).__init__(W=aggregate(Gp.W, Gg.W),
                                               gtype=gtype,
                                               coords=Gg.coords,
                                               plotting=Gg.plotting,
                                               perform_all_checks=False,
                                               **kwargs)
