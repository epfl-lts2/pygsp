# -*- coding: utf-8 -*-

from pygsp.graphs import Graph, Grid2d, ImgPatches


class Grid2dImgPatches(Graph):
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
    aggregate: callable, optional
        Function used for aggregating the weights Wp of the patch graph and the
        weigths Wg 2d grid graph. Default is :func:`lambda Wp, Wg: Wp + Wg`.

    Examples
    --------
    >>> from pygsp import graphs
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::32, ::32])
    >>> G = graphs.Grid2dImgPatches(img)

    """

    def __init__(self, img, patch_shape=(3, 3), n_nbrs=8,
                 aggregate=lambda Wp, Wg: Wp + Wg, **kwargs):
        Gg = Grid2d(shape=img.shape)
        Gp = ImgPatches(img=img, patch_shape=patch_shape, n_nbrs=n_nbrs)
        gtype = '{}_{}'.format(Gg.gtype, Gp.gtype)
        super(Grid2dImgPatches, self).__init__(W=aggregate(Gp.W, Gg.W),
                                               gtype=gtype,
                                               coords=Gg.coords,
                                               plotting=Gg.plotting,
                                               perform_all_checks=False,
                                               **kwargs)
