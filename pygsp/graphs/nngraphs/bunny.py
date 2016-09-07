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

        bunny = PointsCloud("bunny")
        plotting = {"vertex_size": 10, 'vertex_color': (1, 1, 1, 1), 'edge_color': (.5, .5, .5, 1)}

        super(Bunny, self).__init__(Xin=bunny.Xin, epsilon=0.2, NNtype="radius",
                                    plotting=plotting, gtype="Bunny", **kwargs)
