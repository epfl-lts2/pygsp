import numpy as np

from .nngraph import NNGraph  # prevent circular import in Python < 3.5


class Sphere(NNGraph):
    r"""Spherical-shaped graph (NN-graph).

    Parameters
    ----------
    radius : float
        Radius of the sphere (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Sphere(nb_pts=100, seed=42)
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(121)
    >>> ax2 = fig.add_subplot(122, projection='3d')
    >>> _ = ax1.spy(G.W, markersize=1.5)
    >>> _ = _ = G.plot(ax=ax2)

    """

    def __init__(
        self, radius=1, nb_pts=300, nb_dim=3, sampling="random", seed=None, **kwargs
    ):
        self.radius = radius
        self.nb_pts = nb_pts
        self.nb_dim = nb_dim
        self.sampling = sampling
        self.seed = seed

        if self.sampling == "random":
            rs = np.random.RandomState(seed)
            pts = rs.normal(0, 1, (self.nb_pts, self.nb_dim))

            for i in range(self.nb_pts):
                pts[i] /= np.linalg.norm(pts[i])

        else:
            raise ValueError(f"Unknown sampling {sampling}")

        plotting = {
            "vertex_size": 80,
        }

        super().__init__(
            Xin=pts, k=10, center=False, rescale=False, plotting=plotting, **kwargs
        )

    def _get_extra_repr(self):
        attrs = {
            "radius": f"{self.radius:.2f}",
            "nb_pts": self.nb_pts,
            "nb_dim": self.nb_dim,
            "sampling": self.sampling,
            "seed": self.seed,
        }
        attrs.update(super()._get_extra_repr())
        return attrs
