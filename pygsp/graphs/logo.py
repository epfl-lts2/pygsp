# -*- coding: utf-8 -*-

from . import Graph
from pygsp.pointsclouds import PointsCloud

import numpy as np


class Logo(Graph):
    r"""
    Create a graph with the GSP Logo.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Logo()

    """

    def __init__(self):
        logo = PointsCloud("logo")

        self.W = logo.W
        self.coords = logo.coords
        self.info = logo.info

        self.limits = np.array([0, 640, -400, 0])
        self.gtype = 'LogoGSP'

        self.plotting = {"vertex_color": np.array([200./255,
                                                   136./255,
                                                   204./255]),
                         "edge_color": np.array([0, 136./255, 204./255]),
                         "vertex_size": 8}

        super(Logo, self).__init__(plotting=self.plotting, coords=self.coords,
                                   gtype=self.gtype, limits=self.limits,
                                   W=self.W)
