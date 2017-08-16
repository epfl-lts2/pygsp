# -*- coding: utf-8 -*-

import numpy as np

from . import Graph
from ..utils import loadmat


class Logo(Graph):
    r"""
    Create a graph with the GSP logo.

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Logo()

    """

    def __init__(self, **kwargs):

        data = loadmat('pointclouds/logogsp')

        self.info = {"idx_g": data["idx_g"],
                     "idx_s": data["idx_s"],
                     "idx_p": data["idx_p"]}

        plotting = {"limits": np.array([0, 640, -400, 0])}

        super(Logo, self).__init__(W=data['W'], coords=data['coords'],
                                   gtype='LogoGSP', plotting=plotting,
                                   **kwargs)
