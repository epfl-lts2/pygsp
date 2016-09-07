# -*- coding: utf-8 -*-

from . import NNGraph
from pygsp.pointsclouds import PointsCloud

import numpy as np
from math import floor


class TwoMoons(NNGraph):
    r"""
    Create a 2 dimensional graph of the Two Moons.

    Parameters
    ----------
    moontype : string
        You have the freedom to chose if you want to create a standard
        two_moons graph or a synthetised one (default is 'standard').
        'standard' : Create a two_moons graph from a based graph.
        'synthetised' : Create a synthetised two_moon
    sigmag : float
        Variance of the distance kernel (default = 0.05)
    N : int
        Number of vertices (default = 2000)
    sigmad : float
        Variance of the data (do not set it too high or you won't see anything)
        (default = 0.05)
    d : float
        Distance of the two moons (default = 0.5)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G1 = graphs.TwoMoons(moontype='standard')
    >>> G2 = graphs.TwoMoons(moontype='synthetised', N=1000, sigmad=0.1, d=1)

    """

    def create_arc_moon(N, sigmad, d, number):
        phi = np.random.rand(N, 1) * np.pi
        r = 1
        rb = sigmad * np.random.normal(size=(N, 1))
        ab = np.random.rand(N, 1) * 2 * np.pi
        b = rb * np.exp(1j * ab)
        bx = np.real(b)
        by = np.imag(b)

        if number == 1:
            moonx = np.cos(phi) * r + bx + 0.5
            moony = -np.sin(phi) * r + by - (d - 1)/2.
        elif number == 2:
            moonx = np.cos(phi) * r + bx - 0.5
            moony = np.sin(phi) * r + by + (d - 1)/2.

        return np.concatenate((moonx, moony), axis=1)

    def __init__(self, moontype='standard', sigmag=0.05, N=400, sigmad=0.07, d=0.5):

        if moontype == 'standard':
            two_moons = PointsCloud('two_moons')
            Xin = two_moons.Xin

            gtype = 'Two Moons standard'
            self.labels = 2*(np.where(np.arange(1, N + 1).reshape(N, 1) > 1000,
                                      1, 0) + 1)

        elif moontype == 'synthetised':
            gtype = 'Two Moons synthetised'

            N1 = floor(N/2.)
            N2 = N - N1

            # Moon 1
            Coordmoon1 = self.create_arc_moon(N1, sigmad, d, 1)

            # Moon 2
            Coordmoon2 = self.create_arc_moon(N2, sigmad, d, 2)

            Xin = np.concatenate((Coordmoon1, Coordmoon2))
            self.labels = 2*(np.where(np.arange(1, N + 1).reshape(N, 1) >
                                      N1, 1, 0) + 1)

        super(TwoMoons, self).__init__(Xin=Xin, sigma=sigmag, k=5, gtype=gtype)
