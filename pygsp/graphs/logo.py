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

    def __init__(self, **kwargs):
        logo = PointsCloud("logo")

        self.info = logo.info

        plotting = {"vertex_color": np.array([200., 136., 204.]) / 255,
                    "edge_color": np.array([0, 136./255, 204./255]),
                    "limits": np.array([0, 640, -400, 0]),
                    "vertex_size": 8}

        super(Logo, self).__init__(W=logo.W, coords=logo.coords, gtype='LogoGSP',
                                   plotting=plotting, **kwargs)
