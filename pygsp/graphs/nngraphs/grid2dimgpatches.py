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

    See Also
    --------
    ImgPatches
    Grid2d

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.Grid2dImgPatches(img)
    >>> G  # doctest: +ELLIPSIS
    Grid2dImgPatches(n_vertices=64, n_edges=522, N1=8, N2=8, ..., order=0)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, img, aggregate=lambda Wp, Wg: Wp + Wg, **kwargs):

        self.Gg = Grid2d(img.shape[0], img.shape[1])
        self.Gp = ImgPatches(img, **kwargs)

        W = aggregate(self.Gp.W, self.Gg.W)
        super(Grid2dImgPatches, self).__init__(W,
                                               coords=self.Gg.coords,
                                               plotting=self.Gg.plotting)

    def _get_extra_repr(self):
        attrs = self.Gg._get_extra_repr()
        attrs.update(self.Gp._get_extra_repr())
        return attrs
