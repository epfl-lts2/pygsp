# -*- coding: utf-8 -*-

from . import Graph
from ..pointclouds import PointCloud

import numpy as np


class Logo(Graph):
    r"""
    Create a graph with the GSP logo.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Logo()

    """

    def __init__(self, **kwargs):
        logo = PointCloud("logo")

        self.info = logo.info

        plotting = {"limits": np.array([0, 640, -400, 0])}

        super(Logo, self).__init__(W=logo.W, coords=logo.coords, gtype='LogoGSP',
                                   plotting=plotting, **kwargs)
