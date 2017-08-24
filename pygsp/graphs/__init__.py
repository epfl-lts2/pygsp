# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.graphs` module implements the graph class hierarchy. A graph
object is either constructed from an adjacency matrix, or by instantiating one
of the built-in graph models.

The :class:`Graph` base class allows to construct a graph object from any
adjacency matrix and provides a common interface to that object. Derived
classes then allows to instantiate various standard graph models.

**Matrix operators**

* :attr:`Graph.W`: weight matrix
* :attr:`Graph.L`: Laplacian
* :attr:`Graph.U`: Fourier basis
* :attr:`Graph.D`: differential operator

**Checks**

* :meth:`Graph.check_weights`: check the characteristics of the weights matrix
* :meth:`Graph.is_connected`: check the strong connectivity of the input graph
* :meth:`Graph.is_directed`: check if the graph has directed edges

**Attributes computation**

* :meth:`Graph.compute_laplacian`: compute a graph Laplacian
* :meth:`Graph.estimate_lmax`: estimate largest eigenvalue
* :meth:`Graph.compute_fourier_basis`: compute Fourier basis
* :meth:`Graph.compute_differential_operator`: compute differential operator

**Differential operators**

* :meth:`Graph.grad`: compute the gradient of a graph signal
* :meth:`Graph.div`: compute the divergence of a graph signal

**Localization**

* :meth:`Graph.modulate`: generalized modulation operator
* :meth:`Graph.translate`: generalized translation operator

**Transforms** (frequency and vertex-frequency)

* :meth:`Graph.gft`: graph Fourier transform (GFT)
* :meth:`Graph.igft`: inverse graph Fourier transform
* :meth:`Graph.gft_windowed`: windowed GFT
* :meth:`Graph.gft_windowed_gabor`: Gabor windowed GFT
* :meth:`Graph.gft_windowed_normalized`: normalized windowed GFT

**Plotting**

* :meth:`Graph.plot`: plot the graph
* :meth:`Graph.plot_signal`: plot a signal on that graph
* :meth:`Graph.plot_spectrogram`: plot the spectrogram for the graph object

**Others**

* :meth:`Graph.get_edge_list`: return an edge list (alternative representation)
* :meth:`Graph.set_coordinates`: set nodes' coordinates (for plotting)
* :meth:`Graph.subgraph`: create a subgraph
* :meth:`Graph.extract_components`: split the graph into connected components

**Graph models**

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

**Nearest-neighbors graphs constructed from point clouds**

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
