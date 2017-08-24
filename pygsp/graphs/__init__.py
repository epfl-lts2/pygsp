# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.graphs` module implements the graph class hierarchy. A graph
object is either constructed from an adjacency matrix, or by instantiating one
of the built-in graph models.

The :class:`Graph` base class allows to construct a graph object from any
adjacency matrix and provides a common interface to that object.

**Localization**

* :meth:`Graph.modulate`: generalized modulation operator
* :meth:`Graph.translate`: generalized translation operator

**Fourier basis and transforms** (frequency and vertex-frequency)

* :meth:`Graph.gft`: graph Fourier transform (GFT)
* :meth:`Graph.igft`: inverse graph Fourier transform
* :meth:`Graph.gft_windowed`: windowed GFT
* :meth:`Graph.gft_windowed_gabor`: Gabor windowed GFT
* :meth:`Graph.gft_windowed_normalized`: normalized windowed GFT

Derived classes implement various graph models.

* :class:`Airfoil`
* :class:`BarabasiAlbert`
* :class:`Comet`
* :class:`Community`
* :class:`DavidSensorNet`
* :class:`ErdosRenyi`
* :class:`FullConnected`
* :class:`Grid2d`
* :class:`Logo`
* :class:`LowStretchTree`
* :class:`Minnesota`
* :class:`Path`
* :class:`RandomRegular`
* :class:`RandomRing`
* :class:`Ring`
* :class:`Sensor`
* :class:`StochasticBlockModel`
* :class:`SwissRoll`
* :class:`Torus`

Derived classes from :class:`NNGraph` implement nearest-neighbors graphs
constructed from point clouds.

* :class:`Bunny`
* :class:`Cube`
* :class:`ImgPatches`
* :class:`Grid2dImgPatches`
* :class:`Sphere`
* :class:`TwoMoons`

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
