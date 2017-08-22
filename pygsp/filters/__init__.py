# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.filters` module implements methods used for filtering (e.g.
analysis, synthesis, evaluation) and defines commonly used filters that can be
applied to :mod:`pygsp.graphs`. A filter is associated to a graph and is
defined with one or several functions. We define by filterbank a list of
filters, usually centered around different frequencies, applied to a single
graph.

See the :class:`Filter` base class for the documentation of the
interface to the filter object. Derived classes implement various common graph
filters.

**Filterbank of N filters**

* :class:`Abspline`
* :class:`Gabor`
* :class:`HalfCosine`
* :class:`Itersine`
* :class:`MexicanHat`
* :class:`Meyer`
* :class:`SimpleTf`
* :class:`WarpedTranslates`

**Filterbank of 2 filters: low pass and high pass**

* :class:`Regular`
* :class:`Held`
* :class:`Simoncelli`
* :class:`Papadakis`

**Low pass filter**

* :class:`Heat`
* :class:`Expwin`

Moreover, two approximation methods are provided for fast filtering. The
computational complexity of filtering with those approximations is linear with
the number of edges. The complexity of the exact solution, which is to use the
Fourier basis, is quadratic with the number of nodes (without taking into
account the cost of the necessary eigendecomposition of the graph Laplacian).

**Chebyshev polynomials**

* :func:`compute_cheby_coeff`
* :func:`compute_jackson_cheby_coeff`
* :func:`cheby_op`
* :func:`cheby_rect`

**Lanczos algorithm**

* :func:`lanczos`
* :func:`lanczos_op`

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
