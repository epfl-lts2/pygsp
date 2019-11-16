# -*- coding: utf-8 -*-

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

.. _NetworkX: https://networkx.github.io
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

.. _NetworkX's documentation: https://networkx.github.io/documentation/stable/reference/generators.html

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
    
Learning graph from data
------------------------

.. autosummary::

    LearnedGraph

"""

from pygsp import utils as _utils

_GRAPHS = [
    'Graph',
    'Airfoil',
    'BarabasiAlbert',
    'Comet',
    'Community',
    'DavidSensorNet',
    'ErdosRenyi',
    'FullConnected',
    'Grid2d',
    'LineGraph',
    'Logo',
    'LowStretchTree',
    'Minnesota',
    'Path',
    'RandomRegular',
    'RandomRing',
    'Ring',
    'StochasticBlockModel',
    'SwissRoll',
    'Torus'
]
_NNGRAPHS = [
    'NNGraph',
    'Bunny',
    'Cube',
    'ImgPatches',
    'Grid2dImgPatches',
    'Sensor',
    'Sphere',
    'TwoMoons'
]

__all__ = _GRAPHS + _NNGRAPHS

_utils.import_classes(_GRAPHS, 'graphs', 'graphs')
from pygsp.graphs.learned import LearnedFromSmoothSignals
_utils.import_classes(_NNGRAPHS, 'graphs.nngraphs', 'graphs')
