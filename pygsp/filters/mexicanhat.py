# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from math import exp


class MexicanHat(Filter):
    r"""
    Mexican hat Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    t : ndarray
        Vector of scale to be used (Initialized by default at the value of the
        log scale)
    normalize : bool
        Wether to normalize the wavelet by the factor/sqrt(t).
        (default = False)

    Returns
    -------
    out : MexicanHat

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.MexicanHat(G)

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, normalize=False,
                 **kwargs):
        super(MexicanHat, self).__init__(G, **kwargs)

        if t is None:
            G.lmin = G.lmax / lpfactor
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: x * np.exp(-x)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        g = [lambda x: 1.2 * exp(-1) * gl(x / lminfac)]

        for i in range(Nf - 1):
            if normalize:
                g.append(lambda x, ind=i: np.sqrt(t[ind]) *
                         gb(self.t[ind] * x))
            else:
                g.append(lambda x, ind=i: gb(self.t[ind] * x))

        self.g = g
