# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.graphs` module implements the graph class hierarchy. A graph
object is either constructed from an adjacency matrix, or by instantiating one
of the built-in graph models.

The :class:`pygsp.graphs.Graph` base class allows to construct a graph object
from any adjacency matrix and provides a common interface to that object.

Derived classes implement various graph models.

* :class:`pygsp.graphs.Airfoil`
* :class:`pygsp.graphs.BarabasiAlbert`
* :class:`pygsp.graphs.Comet`
* :class:`pygsp.graphs.Community`
* :class:`pygsp.graphs.DavidSensorNet`
* :class:`pygsp.graphs.ErdosRenyi`
* :class:`pygsp.graphs.FullConnected`
* :class:`pygsp.graphs.Grid2d`
* :class:`pygsp.graphs.Logo`
* :class:`pygsp.graphs.LowStretchTree`
* :class:`pygsp.graphs.Minnesota`
* :class:`pygsp.graphs.Path`
* :class:`pygsp.graphs.RandomRegular`
* :class:`pygsp.graphs.RandomRing`
* :class:`pygsp.graphs.Ring`
* :class:`pygsp.graphs.Sensor`
* :class:`pygsp.graphs.StochasticBlockModel`
* :class:`pygsp.graphs.SwissRoll`
* :class:`pygsp.graphs.Torus`

Derived classes from :class:`pygsp.graphs.NNGraph` implement nearest-neighbors
graphs constructed from point clouds.

* :class:`pygsp.graphs.Bunny`
* :class:`pygsp.graphs.Cube`
* :class:`pygsp.graphs.ImgPatches`
* :class:`pygsp.graphs.Grid2dImgPatches`
* :class:`pygsp.graphs.Sphere`
* :class:`pygsp.graphs.TwoMoons`

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
