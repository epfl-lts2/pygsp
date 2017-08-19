# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.operators` module implements some operators on graphs.

**Differential operators**

* :func:`pygsp.operators.grad`: compute the gradient of a graph signal
* :func:`pygsp.operators.div`: compute the divergence of a graph signal

**Transforms** (frequency and vertex-frequency)

* :func:`pygsp.operators.gft`: graph Fourier transform
* :func:`pygsp.operators.igft`: inverse graph Fourier transform
* :func:`pygsp.operators.generalized_wft`: graph windowed Fourier transform
* :func:`pygsp.operators.gabor_wft`: graph windowed Fourier transform
* :func:`pygsp.operators.ngwft`: normalized graph windowed Fourier transform

**Localization**

* :func:`pygsp.operators.localize`: localize a kernel
* :func:`pygsp.operators.modulate`: generalized modulation operator
* :func:`pygsp.operators.translate`: generalized translation operator

**Reduction** Functionalities for the reduction of graphs' vertex set while keeping the graph structure.

* :func:`pygsp.operators.tree_multiresolution`: compute a multiresolution of trees
* :func:`pygsp.operators.graph_multiresolution`: compute a pyramid of graphs
* :func:`pygsp.operators.kron_reduction`: compute the Kron reduction
* :func:`pygsp.operators.pyramid_analysis`: analysis operator for graph pyramid
* :func:`pygsp.operators.pyramid_synthesis`: synthesis operator for graph pyramid
* :func:`pygsp.operators.pyramid_cell2coeff`: keep only the necessary coefficients
* :func:`pygsp.operators.interpolate`: interpolate a signal
* :func:`pygsp.operators.graph_sparsify`: sparsify a graph

"""

from pygsp import utils as _utils

_DIFFERENCE = [
    'grad',
    'div',
]
_TRANSFORMS = [
    'gft',
    'igft',
    'generalized_wft',
    'gabor_wft',
    'ngwft',
]
_LOCALIZATION = [
    'localize',
    'modulate',
    'translate',
]
_REDUCTION = [
    'tree_multiresolution',
    'graph_multiresolution',
    'kron_reduction',
    'pyramid_analysis',
    'pyramid_synthesis',
    'pyramid_cell2coeff',
    'interpolate',
    'graph_sparsify',
]

__all__ = _DIFFERENCE + _TRANSFORMS + _LOCALIZATION + _REDUCTION

_utils.import_functions(_DIFFERENCE, 'operators.difference', 'operators')
_utils.import_functions(_TRANSFORMS, 'operators.transforms', 'operators')
_utils.import_functions(_LOCALIZATION, 'operators.localization', 'operators')
_utils.import_functions(_REDUCTION, 'operators.reduction', 'operators')
