# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class TwoMoons(NNGraph):
    r"""Two Moons (NN-graph).

    Parameters
    ----------
    moontype : 'standard' or 'synthesized'
        You have the freedom to chose if you want to create a standard
        two_moons graph or a synthesized one (default is 'standard').
        'standard' : Create a two_moons graph from a based graph.
        'synthesized' : Create a synthesized two_moon
    kernel_width : float
        Variance of the distance kernel (default = 0.05)
    dim : int
        The dimensionality of the points (default = 2).
        Only valid for moontype == 'standard'.
    N : int
        Number of vertices (default = 2000)
        Only valid for moontype == 'synthesized'.
    variance : float
        Variance of the data (do not set it too high or you won't see anything)
        (default = 0.05)
        Only valid for moontype == 'synthesized'.
    distance : float
        Distance between the two moons (default = 0.5)
        Only valid for moontype == 'synthesized'.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.TwoMoons()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> _ = G.plot(edges=True, ax=axes[1])

    """

    def _create_arc_moon(self, N, variance, distance, number, seed):
        rs = np.random.RandomState(seed)
        phi = rs.rand(N, 1) * np.pi
        r = 1
        rb = variance * rs.normal(size=(N, 1))
        ab = rs.rand(N, 1) * 2 * np.pi
        b = rb * np.exp(1j * ab)
        bx = np.real(b)
        by = np.imag(b)

        if number == 1:
            moonx = np.cos(phi) * r + bx + 0.5
            moony = -np.sin(phi) * r + by - (distance - 1)/2.
        elif number == 2:
            moonx = np.cos(phi) * r + bx - 0.5
            moony = np.sin(phi) * r + by + (distance - 1)/2.

        return np.concatenate((moonx, moony), axis=1)

    def __init__(self, moontype='standard', dim=2, kernel_width=0.05,
                 N=400, variance=0.07, distance=0.5, seed=None, **kwargs):

        self.moontype = moontype
        self.dim = dim
        self.kernel_width = kernel_width
        self.variance = variance
        self.distance = distance
        self.seed = seed

        if moontype == 'standard':
            N1, N2 = 1000, 1000
            data = utils.loadmat('pointclouds/two_moons')
            coords = data['features'][:dim].T

        elif moontype == 'synthesized':
            N1 = N // 2
            N2 = N - N1

            coords1 = self._create_arc_moon(N1, variance, distance, 1, seed)
            coords2 = self._create_arc_moon(N2, variance, distance, 2, seed)

            coords = np.concatenate((coords1, coords2))

        else:
            raise ValueError('Unknown moontype {}'.format(moontype))

        self.labels = np.concatenate((np.zeros(N1), np.ones(N2)))

        plotting = {
            'vertex_size': 30,
        }

        super(TwoMoons, self).__init__(coords, kernel_width=kernel_width, k=5,
                                       plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        attrs = {'moontype': self.moontype,
                 'dim': self.dim,
                 'kernel_width': '{:.2f}'.format(self.kernel_width),
                 'variance': '{:.2f}'.format(self.variance),
                 'distance': '{:.2f}'.format(self.distance),
                 'seed': self.seed}
        attrs.update(super(TwoMoons, self)._get_extra_repr())
        return attrs
