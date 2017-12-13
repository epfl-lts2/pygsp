# -*- coding: utf-8 -*-

from pygsp import utils
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class ImgPatches(NNGraph):
    r"""NN-graph between patches of an image.

    Parameters
    ----------
    img : array
        Input image.
    patch_shape : tuple, optional
        Dimensions of the patch window. Syntax: (height, width), or (height,),
        in which case width = height.
    kwargs : dict
        Parameters passed to :class:`NNGraph`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.ImgPatches(img)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, img, patch_shape=(3, 3), **kwargs):

        self.img = img
        X = utils.extract_patches(img, patch_shape=patch_shape)

        super(ImgPatches, self).__init__(X, gtype='patch-graph', **kwargs)
