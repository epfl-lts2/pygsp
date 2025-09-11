from pygsp import utils

from .nngraph import NNGraph  # prevent circular import in Python < 3.5


class Bunny(NNGraph):
    r"""Stanford bunny (NN-graph).

    References
    ----------
    See :cite:`turk1994zippered`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Bunny()
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=0.1)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(self, **kwargs):
        data = utils.loadmat("pointclouds/bunny")

        plotting = {
            "vertex_size": 10,
            "elevation": -90,
            "azimuth": 90,
            "distance": 8,
        }

        super().__init__(
            Xin=data["bunny"],
            epsilon=0.02,
            NNtype="radius",
            center=False,
            rescale=False,
            plotting=plotting,
            **kwargs,
        )
