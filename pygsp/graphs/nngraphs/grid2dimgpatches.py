# -*- coding: utf-8 -*-

# prevent circular import in Python < 3.5
from pygsp.graphs import Graph, Grid2d, ImgPatches


class Grid2dImgPatches(Graph):
    r"""Union of a patch graph with a 2D grid graph.

    Parameters
    ----------
    img : array
        Input image.
    aggregate: callable, optional
        Function to aggregate the weights ``Wp`` of the patch graph and the
        ``Wg`` of the grid graph. Default is ``lambda Wp, Wg: Wp + Wg``.
    kwargs : dict
        Parameters passed to :class:`ImgPatches`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.Grid2dImgPatches(img)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, img, aggregate=lambda Wp, Wg: Wp + Wg, **kwargs):

        Gg = Grid2d(img.shape[0], img.shape[1])
        Gp = ImgPatches(img, **kwargs)

        gtype = '{}_{}'.format(Gg.gtype, Gp.gtype)

        super(Grid2dImgPatches, self).__init__(W=aggregate(Gp.W, Gg.W),
                                               gtype=gtype,
                                               coords=Gg.coords,
                                               plotting=Gg.plotting)
