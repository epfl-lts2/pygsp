# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.operators` module implements some operators on graphs.

**Differential operators**

* :func:`grad`: compute the gradient of a graph signal
* :func:`div`: compute the divergence of a graph signal

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

__all__ = _DIFFERENCE + _REDUCTION

_utils.import_functions(_DIFFERENCE, 'operators.difference', 'operators')
_utils.import_functions(_REDUCTION, 'operators.reduction', 'operators')
