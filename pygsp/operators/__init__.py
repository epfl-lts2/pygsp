# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.operators` module implements some operators on graphs.

**Differential operators**

* :func:`grad`: compute the gradient of a graph signal
* :func:`div`: compute the divergence of a graph signal

**Transforms** (frequency and vertex-frequency)

* :func:`gft`: graph Fourier transform (GFT)
* :func:`igft`: inverse graph Fourier transform
* :func:`gft_windowed`: windowed GFT
* :func:`gft_windowed_gabor`: Gabor windowed GFT
* :func:`gft_windowed_normalized`: normalized windowed GFT

**Localization**

* :func:`modulate`: generalized modulation operator
* :func:`translate`: generalized translation operator

**Reduction** Functionalities for the reduction of graphs' vertex set while keeping the graph structure.

* :func:`tree_multiresolution`: compute a multiresolution of trees
* :func:`graph_multiresolution`: compute a pyramid of graphs
* :func:`kron_reduction`: compute the Kron reduction
* :func:`pyramid_analysis`: analysis operator for graph pyramid
* :func:`pyramid_synthesis`: synthesis operator for graph pyramid
* :func:`pyramid_cell2coeff`: keep only the necessary coefficients
* :func:`interpolate`: interpolate a signal
* :func:`graph_sparsify`: sparsify a graph

"""

from pygsp import utils as _utils

_DIFFERENCE = [
    'grad',
    'div',
]
_TRANSFORMS = [
    'gft',
    'igft',
    'gft_windowed',
    'gft_windowed_gabor',
    'gft_windowed_normalized',
]
_LOCALIZATION = [
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
