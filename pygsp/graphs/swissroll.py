# -*- coding: utf-8 -*-

from . import Graph
import numpy as np
from math import sqrt, pi
from pygsp import utils


class SwissRoll(Graph):
    r"""
    Create a swiss roll graph.

    Parameters
    ----------
    N : int
        Number of vertices (default = 400)
    a : int
        (default = 1)
    b : int
        (default = 4)
    dim : int
        (default = 3)
    thresh : float
        (default = 1e-6)
    s : float
        sigma (default =  sqrt(2./N))
    noise : bool
        Wether to add noise or not (default = False)
    srtype : str
        Swiss roll Type, possible arguments are 'uniform' or 'classic'
        (default = 'uniform')

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.SwissRoll()

    """

    def __init__(self, N=400, a=1, b=4, dim=3, thresh=1e-6, s=None,
                 noise=False, srtype='uniform'):

        from pygsp import plotting

        self.dim = dim
        self.N = N
        if s is None:
            s = sqrt(2./N)

        y1 = np.random.rand(N)
        y2 = np.random.rand(N)
        if srtype == 'uniform':
            tt = np.sqrt((b * b - a * a) * y1 + a * a)
        elif srtype == 'classic':
            tt = (b - a) * y1 + a
        self.gtype = 'swiss roll' + srtype
        tt *= pi
        h = 21 * y2
        if dim == 2:
            x = np.array((tt*np.cos(tt), tt * np.sin(tt)))
        elif dim == 3:
            x = np.array((tt*np.cos(tt), h, tt * np.sin(tt)))

        if noise:
            x += np.random.randn(*x.shape)

        self.x = x

        self.limits = np.array([-1, 1, -1, 1, -1, 1])

        coords = plotting.rescale_center(x)
        dist = utils.distanz(coords)
        W = np.exp(-np.power(dist, 2) / (2. * s**2))
        W -= np.diag(np.diag(W))
        W = np.where(W < thresh, 0, W)

        self.W = W

        self.coords = coords.T
        super(SwissRoll, self).__init__(W=self.W, coords=self.coords,
                                        limits=self.limits, gtype=self.gtype)
