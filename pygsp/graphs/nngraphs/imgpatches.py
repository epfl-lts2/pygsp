# -*- coding: utf-8 -*-

import numpy as np

from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class ImgPatches(NNGraph):
    r"""NN-graph between patches of an image.

    Extract a feature vector in the form of a patch for every pixel of an
    image, then construct a nearest-neighbor graph between these feature
    vectors. The feature matrix, i.e. the patches, can be found in :attr:`Xin`.

    Parameters
    ----------
    img : array
        Input image.
    patch_shape : tuple, optional
        Dimensions of the patch window. Syntax: (height, width), or (height,),
        in which case width = height.
    kwargs : dict
        Parameters passed to :class:`NNGraph`.

    Notes
    -----
    The feature vector of a pixel `i` will consist of the stacking of the
    intensity values of all pixels in the patch centered at `i`, for all color
    channels. So, if the input image has `d` color channels, the dimension of
    the feature vector of each pixel is (patch_shape[0] * patch_shape[1] * d).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.ImgPatches(img, patch_shape=(3, 3))
    >>> print('{} nodes ({} x {} pixels)'.format(G.Xin.shape[0], *img.shape))
    64 nodes (8 x 8 pixels)
    >>> print('{} features per node'.format(G.Xin.shape[1]))
    9 features per node
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, img, patch_shape=(3, 3), **kwargs):

        self.img = img

        try:
            h, w, d = img.shape
        except ValueError:
            try:
                h, w = img.shape
                d = 0
            except ValueError:
                print("Image should be at least a 2D array.")

        try:
            r, c = patch_shape
        except ValueError:
            r = patch_shape[0]
            c = r

        pad_width = [(int((r - 0.5) / 2.), int((r + 0.5) / 2.)),
                     (int((c - 0.5) / 2.), int((c + 0.5) / 2.))]

        if d == 0:
            window_shape = (r, c)
            d = 1  # For the reshape in the return call
        else:
            pad_width += [(0, 0)]
            window_shape = (r, c, d)

        # Pad the image.
        img = np.pad(img, pad_width=pad_width, mode='symmetric')

        # Extract patches as node features.
        # Alternative: sklearn.feature_extraction.image.extract_patches_2d.
        #              sklearn has much less dependencies than skimage.
        try:
            import skimage
        except Exception:
            raise ImportError('Cannot import skimage, which is needed to '
                              'extract patches. Try to install it with '
                              'pip (or conda) install scikit-image.')
        patches = skimage.util.view_as_windows(img, window_shape=window_shape)
        patches = patches.reshape((h * w, r * c * d))

        super(ImgPatches, self).__init__(patches,
                                         gtype='patch-graph',
                                         **kwargs)
