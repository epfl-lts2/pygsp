# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Logo(Graph):
    r"""GSP logo.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Logo()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(ax=axes[1])

    """

    def __init__(self, **kwargs):

        data = utils.loadmat('pointclouds/logogsp')

        self.info = {"idx_g": data["idx_g"],
                     "idx_s": data["idx_s"],
                     "idx_p": data["idx_p"]}

        plotting = {"limits": np.array([0, 640, -400, 0])}

        super(Logo, self).__init__(W=data['W'], coords=data['coords'],
                                   gtype='LogoGSP', plotting=plotting,
                                   **kwargs)
