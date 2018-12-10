# -*- coding: utf-8 -*-

# prevent circular import in Python < 3.5
from .stochasticblockmodel import StochasticBlockModel


class ErdosRenyi(StochasticBlockModel):
    r"""Erdos Renyi graph.

    The Erdos Renyi graph is constructed by randomly connecting nodes. Each
    edge is included in the graph with probability p, independently from any
    other edge. All edge weights are equal to 1.

    Parameters
    ----------
    N : int
        Number of nodes (default is 100).
    p : float
        Probability to connect a node with another one.
    directed : bool
        Allow directed edges if True (default is False).
    self_loops : bool
        Allow self loops if True (default is False).
    connected : bool
        Force the graph to be connected (default is False).
    n_try : int
        Maximum number of trials to get a connected graph (default is 10).
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.ErdosRenyi(N=64, seed=42)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(self, N=100, p=0.1, directed=False, self_loops=False,
                 connected=False, n_try=10, seed=None, **kwargs):

        super(ErdosRenyi, self).__init__(N=N, k=1, p=p,
                                         directed=directed,
                                         self_loops=self_loops,
                                         connected=connected,
                                         n_try=n_try,
                                         seed=seed,
                                         **kwargs)
