# -*- coding: utf-8 -*-

r"""
This module implements filters and contains predefined filters that can be directly applied to graphs.

A filter is associated to a graph and is defined with one or several function(s).
We define by Filterbank a list of filters applied to a single graph.
Tools for the analysis, the synthesis and the evaluation are provided to work with the filters on the graphs.
For specific information, :ref:`see details here<filters-api>`.
"""

from pygsp import utils as _utils

_FILTERS = [
    'Filter',
    'Abspline',
    'Expwin',
    'Gabor',
    'HalfCosine',
    'Heat',
    'Held',
    'Itersine',
    'MexicanHat',
    'Meyer',
    'Papadakis',
    'Regular',
    'Simoncelli',
    'SimpleTf',
    'WarpedTranslates'
]
_APPROXIMATIONS = [
    'compute_cheby_coeff',
    'compute_jackson_cheby_coeff',
    'cheby_op',
    'cheby_rect',
    'lanczos',
    'lanczos_op'
]

__all__ = _FILTERS + _APPROXIMATIONS

_utils.import_classes(_FILTERS, 'filters', 'filters')
_utils.import_functions(_APPROXIMATIONS, 'filters.approximations', 'filters')
