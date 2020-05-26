# -*- coding: utf-8 -*-

from . import Comet  # prevent circular import in Python < 3.5


class Star(Comet):
    r"""Star graph.

    A star with a central vertex and `N-1` branches.
    The central vertex has degree `N-1`, the others have degree 1.

    Parameters
    ----------
    N : int
        Number of vertices.

    See Also
    --------
    Comet : Generalization with a longer branch as a tail

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> graph = graphs.Star(15)
    >>> graph
    Star(n_vertices=15, n_edges=14)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(graph.W)
    >>> _ = graph.plot(ax=axes[1])

    """

    def __init__(self, N=10, **kwargs):
        plotting = dict(limits=[-1.1, 1.1, -1.1, 1.1])
        plotting.update(kwargs.get('plotting', {}))
        super(Star, self).__init__(N, N-1, plotting=plotting, **kwargs)

    def _get_extra_repr(self):
        return dict()  # Suppress Comet repr.
