# -*- coding: utf-8 -*-

from . import NNGraph
from pygsp.pointsclouds import PointsCloud


class Bunny(NNGraph):
    r"""
    Create a graph of the stanford bunny.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Bunny()

    References
    ----------
    :cite:`turk1994zippered`

    """

    def __init__(self, **kwargs):

        self.NNtype = "radius"
        self.rescale = True
        self.center = True
        self.epsilon = 0.2
        self.gtype = "Bunny"

        bunny = PointsCloud("bunny")
        self.Xin = bunny.Xin

        self.plotting = {"vertex_size": 10}

        super(Bunny, self).__init__(Xin=self.Xin, center=self.center,
                                    rescale=self.rescale, epsilon=self.epsilon,
                                    plotting=self.plotting, NNtype=self.NNtype,
                                    gtype=self.gtype, **kwargs)
