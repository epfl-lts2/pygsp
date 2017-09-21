# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class SwissRoll(Graph):
    r"""Sampled Swiss roll manifold.

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
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.SwissRoll(N=200, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1)
    >>> G.plot(ax=ax2)

    """

    def __init__(self, N=400, a=1, b=4, dim=3, thresh=1e-6, s=None,
                 noise=False, srtype='uniform', seed=None, **kwargs):

        if s is None:
            s = np.sqrt(2. / N)

        rs = np.random.RandomState(seed)
        y1 = rs.rand(N)
        y2 = rs.rand(N)

        if srtype == 'uniform':
            tt = np.sqrt((b * b - a * a) * y1 + a * a)
        elif srtype == 'classic':
            tt = (b - a) * y1 + a
        tt *= np.pi

        if dim == 2:
            x = np.array((tt * np.cos(tt), tt * np.sin(tt)))
        elif dim == 3:
            x = np.array((tt * np.cos(tt), 21 * y2, tt * np.sin(tt)))

        if noise:
            x += rs.randn(*x.shape)

        self.x = x
        self.dim = dim

        coords = utils.rescale_center(x)
        dist = utils.distanz(coords)
        W = np.exp(-np.power(dist, 2) / (2. * s**2))
        W -= np.diag(np.diag(W))
        W[W < thresh] = 0

        plotting = {
            'vertex_size': 60,
            'limits': np.array([-1, 1, -1, 1, -1, 1]),
            'elevation': 15,
            'azimuth': -90,
            'distance': 7,
        }
        gtype = 'swiss roll {}'.format(srtype)

        super(SwissRoll, self).__init__(W=W, coords=coords.T,
                                        plotting=plotting, gtype=gtype,
                                        **kwargs)
