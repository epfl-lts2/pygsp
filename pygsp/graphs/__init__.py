r"""
The :mod:`pygsp.graphs` module implements the graph class hierarchy. A graph
object is either constructed from an adjacency matrix, or by instantiating one
of the built-in graph models.

Interface
=========

The :class:`Graph` base class allows to construct a graph object from any
adjacency matrix and provides a common interface to that object. Derived
classes then allows to instantiate various standard graph models.

Attributes
----------

**Matrix operators**

.. autosummary::

    Graph.W
    Graph.L
    Graph.U
    Graph.D

**Vectors**

.. autosummary::

    Graph.d
    Graph.dw
    Graph.e

**Scalars**

.. autosummary::

    Graph.lmax
    Graph.coherence

Attributes computation
----------------------

.. autosummary::

    Graph.compute_laplacian
    Graph.estimate_lmax
    Graph.compute_fourier_basis
    Graph.compute_differential_operator

Differential operators
----------------------

.. autosummary::

    Graph.grad
    Graph.div
    Graph.dirichlet_energy

Transforms
----------

.. autosummary::

    Graph.gft
    Graph.igft

Vertex-frequency transforms are implemented as filter banks and are found in
:mod:`pygsp.filters` (such as :class:`~pygsp.filters.Gabor` and
:class:`~pygsp.filters.Modulation`).

Checks
------

.. autosummary::

    Graph.is_weighted
    Graph.is_connected
    Graph.is_directed
    Graph.has_loops

Plotting
--------

.. autosummary::

    Graph.plot
    Graph.plot_spectrogram

Import and export (I/O)
-----------------------

We provide import and export facility to two well-known Python packages for
network analysis: NetworkX_ and graph-tool_.
Those packages and the PyGSP are fundamentally different in their goals (graph
analysis versus graph signal analysis) and graph representations (if in the
PyGSP everything is an ndarray, in NetworkX everything is a dictionary).
Those tools are complementary and good interoperability is necessary to exploit
the strengths of each tool.
We ourselves leverage NetworkX and graph-tool to save and load graphs.

Note: to tie a signal with the graph, such that they are exported together,
attach it first with :meth:`Graph.set_signal`.

.. _NetworkX: https://networkx.org
.. _graph-tool: https://graph-tool.skewed.de

.. autosummary::

    Graph.load
    Graph.save
    Graph.from_networkx
    Graph.to_networkx
    Graph.from_graphtool
    Graph.to_graphtool

Others
------

.. autosummary::

    Graph.get_edge_list
    Graph.set_signal
    Graph.set_coordinates
    Graph.subgraph
    Graph.extract_components

Graph models
============

In addition to the below graphs, useful resources are the random graph
generators from NetworkX (see `NetworkX's documentation`_) and graph-tool (see
:mod:`graph_tool.generation`), as well as graph-tool's assortment of standard
networks (see :mod:`graph_tool.collection`).
Any graph created by NetworkX or graph-tool can be imported in the PyGSP with
:meth:`Graph.from_networkx` and :meth:`Graph.from_graphtool`.

.. _NetworkX's documentation: https://networkx.org/documentation/stable/reference/generators.html

Graphs built from other graphs
------------------------------

.. autosummary::

    LineGraph

Generated graphs
----------------

.. autosummary::

    Airfoil
    BarabasiAlbert
    Comet
    Community
    DavidSensorNet
    ErdosRenyi
    FullConnected
    Grid2d
    Logo
    LowStretchTree
    Minnesota
    Path
    RandomRegular
    RandomRing
    Ring
    Star
    StochasticBlockModel
    SwissRoll
    Torus

Nearest-neighbors graphs constructed from point clouds
------------------------------------------------------

.. autosummary::

    NNGraph
    Bunny
    Cube
    ImgPatches
    Grid2dImgPatches
    Sensor
    Sphere
    TwoMoons

"""

from .airfoil import Airfoil  # noqa: F401
from .barabasialbert import BarabasiAlbert  # noqa: F401
from .comet import Comet  # noqa: F401
from .community import Community  # noqa: F401
from .davidsensornet import DavidSensorNet  # noqa: F401
from .erdosrenyi import ErdosRenyi  # noqa: F401
from .fullconnected import FullConnected  # noqa: F401
from .graph import Graph  # noqa: F401
from .grid2d import Grid2d  # noqa: F401
from .linegraph import LineGraph  # noqa: F401
from .logo import Logo  # noqa: F401
from .lowstretchtree import LowStretchTree  # noqa: F401
from .minnesota import Minnesota  # noqa: F401
from .nngraphs.bunny import Bunny  # noqa: F401
from .nngraphs.cube import Cube  # noqa: F401
from .nngraphs.grid2dimgpatches import Grid2dImgPatches  # noqa: F401
from .nngraphs.imgpatches import ImgPatches  # noqa: F401
from .nngraphs.nngraph import NNGraph  # noqa: F401
from .nngraphs.sensor import Sensor  # noqa: F401
from .nngraphs.sphere import Sphere  # noqa: F401
from .nngraphs.twomoons import TwoMoons  # noqa: F401
from .path import Path  # noqa: F401
from .randomregular import RandomRegular  # noqa: F401
from .randomring import RandomRing  # noqa: F401
from .ring import Ring  # noqa: F401
from .star import Star  # noqa: F401
from .stochasticblockmodel import StochasticBlockModel  # noqa: F401
from .swissroll import SwissRoll  # noqa: F401
from .torus import Torus  # noqa: F401
