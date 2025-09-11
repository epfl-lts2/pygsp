import numpy as np
from scipy import sparse

from pygsp import utils

from .graph import Graph  # prevent circular import in Python < 3.5

logger = utils.build_logger(__name__)


class LineGraph(Graph):
    r"""Build the line graph of a graph.

    Each vertex of the line graph represents an edge in the original graph. Two
    vertices are connected if the edges they represent share a vertex in the
    original graph.

    Parameters
    ----------
    graph : :class:`Graph`

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> graph = graphs.Sensor(5, k=2, seed=10)
    >>> line_graph = graphs.LineGraph(graph)
    >>> fig, ax = plt.subplots()
    >>> fig, ax = graph.plot('blue', edge_color='blue', indices=True, ax=ax)
    >>> fig, ax = line_graph.plot('red', edge_color='red', indices=True, ax=ax)
    >>> _ = ax.set_title('graph and its line graph')

    """

    def __init__(self, graph, **kwargs):
        if graph.is_weighted():
            logger.warning(
                "Your graph is weighted, and is considered "
                "unweighted to build a binary line graph."
            )

        graph.compute_differential_operator()
        # incidence = np.abs(graph.D)  # weighted?
        incidence = graph.D != 0

        adjacency = incidence.T.dot(incidence).astype(int)
        adjacency -= sparse.identity(graph.n_edges, dtype=int)

        try:
            coords = incidence.T.dot(graph.coords) / 2
        except AttributeError:
            coords = None

        super().__init__(adjacency, coords=coords, plotting=graph.plotting, **kwargs)
