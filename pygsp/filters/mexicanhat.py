# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


class MexicanHat(Filter):
    r"""
    Mexican hat filterbank

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    scales : ndarray
        Vector of scales to be used.
        By default, initialized with :func:`pygsp.utils.compute_log_scales`.
    normalize : bool
        Wether to normalize the wavelet by the factor/sqrt(t).
        (default = False)

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.MexicanHat(G)

    """

    def __init__(self, G, Nf=6, lpfactor=20, scales=None, normalize=False,
                 **kwargs):

        G.lmin = G.lmax / lpfactor

        if scales is None:
            self.scales = utils.compute_log_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.scales = scales

        gb = lambda x: x * np.exp(-x)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        g = [lambda x: 1.2 * np.exp(-1) * gl(x / lminfac)]

        for i in range(Nf - 1):
            if normalize:
                g.append(lambda x, ind=i: np.sqrt(self.scales[ind]) *
                         gb(self.scales[ind] * x))
            else:
                g.append(lambda x, ind=i: gb(self.scales[ind] * x))

        super(MexicanHat, self).__init__(G, g, **kwargs)
