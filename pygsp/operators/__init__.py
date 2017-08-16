# -*- coding: utf-8 -*-

from pygsp import utils as _utils

_DIFFERENCE = [
    'grad_mat',
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
