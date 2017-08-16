# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.filters` module implements methods used for filtering (e.g.
analysis, synthesis, evaluation) and defines commonly used filters that can be
applied to :mod:`pygsp.graphs`. A filter is associated to a graph and is
defined with one or several functions. We define by filterbank a list of
filters, usually centered around different frequencies, applied to a single
graph.

See the :class:`pygsp.filters.Filter` base class for the documentation of the
interface to the filter object. Derived classes implement various common graph
filters.

**Filterbank of N filters**

* :class:`pygsp.filters.Abspline`
* :class:`pygsp.filters.Gabor`
* :class:`pygsp.filters.HalfCosine`
* :class:`pygsp.filters.Itersine`
* :class:`pygsp.filters.MexicanHat`
* :class:`pygsp.filters.Meyer`
* :class:`pygsp.filters.SimpleTf`
* :class:`pygsp.filters.WarpedTranslates`

**Filterbank of 2 filters: low pass and high pass**

* :class:`pygsp.filters.Regular`
* :class:`pygsp.filters.Held`
* :class:`pygsp.filters.Simoncelli`
* :class:`pygsp.filters.Papadakis`

**Low pass filter**

* :class:`pygsp.filters.Heat`
* :class:`pygsp.filters.Expwin`

Moreover, two approximation methods are provided for fast filtering. The
computational complexity of filtering with those approximations is linear with
the number of edges. The complexity of the exact solution, which is to use the
Fourier basis, is quadratic with the number of nodes (without taking into
account the cost of the necessary eigendecomposition of the graph Laplacian).

**Chebyshev polynomials**

* :class:`pygsp.filters.compute_cheby_coeff`
* :class:`pygsp.filters.compute_jackson_cheby_coeff`
* :class:`pygsp.filters.cheby_op`
* :class:`pygsp.filters.cheby_rect`

**Lanczos algorithm**

* :class:`pygsp.filters.lanczos`
* :class:`pygsp.filters.lanczos_op`

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
