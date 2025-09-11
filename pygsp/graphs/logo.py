import numpy as np

from pygsp import utils

from .graph import Graph  # prevent circular import in Python < 3.5


class Logo(Graph):
    r"""GSP logo.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Logo()
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, **kwargs):
        data = utils.loadmat("pointclouds/logogsp")

        # Remove 1 because the index in python start at 0 and not at 1
        self.info = {
            "idx_g": data["idx_g"] - 1,
            "idx_s": data["idx_s"] - 1,
            "idx_p": data["idx_p"] - 1,
        }

        plotting = {"limits": np.array([0, 640, -400, 0])}

        super().__init__(data["W"], coords=data["coords"], plotting=plotting, **kwargs)
