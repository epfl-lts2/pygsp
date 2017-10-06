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

Matrix operators
----------------

.. autosummary::

    Graph.W
    Graph.L
    Graph.U
    Graph.D

Checks
------

.. autosummary::

    Graph.check_weights
    Graph.is_connected
    Graph.is_directed

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

Localization
------------

.. autosummary::

    Graph.modulate
    Graph.translate

Transforms (frequency and vertex-frequency)
-------------------------------------------

.. autosummary::

    Graph.gft
    Graph.igft
    Graph.gft_windowed
    Graph.gft_windowed_gabor
    Graph.gft_windowed_normalized

Plotting
--------

.. autosummary::

    Graph.plot
    Graph.plot_signal
    Graph.plot_spectrogram

Others
------

.. autosummary::

    Graph.get_edge_list
    Graph.set_coordinates
    Graph.subgraph
    Graph.extract_components

Graph models
============

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
    Sensor
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
    Sphere
    TwoMoons

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
    'Logo',
    'LowStretchTree',
    'Minnesota',
    'Path',
    'RandomRegular',
    'RandomRing',
    'Ring',
    'Sensor',
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
    'Sphere',
    'TwoMoons'
]

__all__ = _GRAPHS + _NNGRAPHS

_utils.import_classes(_GRAPHS, 'graphs', 'graphs')
_utils.import_classes(_NNGRAPHS, 'graphs.nngraphs', 'graphs')
