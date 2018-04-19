# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.filters` module implements methods used for filtering and
defines commonly used filters that can be applied to :mod:`pygsp.graphs`. A
filter is associated to a graph and is defined with one or several functions.
We define by filter bank a list of filters, usually centered around different
frequencies, applied to a single graph.

Interface
---------

The :class:`Filter` base class implements a common interface to all filters:

.. autosummary::

    Filter.evaluate
    Filter.filter
    Filter.analyze
    Filter.synthesize
    Filter.compute_frame
    Filter.estimate_frame_bounds
    Filter.plot
    Filter.localize

Filters
-------

Then, derived classes implement various common graph filters.

**Filter banks of N filters**

.. autosummary::

    Abspline
    Gabor
    HalfCosine
    Itersine
    MexicanHat
    Meyer
    SimpleTight

**Filter banks of 2 filters: a low pass and a high pass**

.. autosummary::

    Regular
    Held
    Simoncelli
    Papadakis

**Low pass filters**

.. autosummary::

    Heat
    Expwin

Approximations
--------------

Moreover, two approximation methods are provided for fast filtering. The
computational complexity of filtering with those approximations is linear with
the number of edges. The complexity of the exact solution, which is to use the
Fourier basis, is quadratic with the number of nodes (without taking into
account the cost of the necessary eigendecomposition of the graph Laplacian).

**Chebyshev polynomials**

.. autosummary::

    compute_cheby_coeff
    compute_jackson_cheby_coeff
    cheby_op
    cheby_rect

**Lanczos algorithm**

.. autosummary::

    lanczos
    lanczos_op

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
    'SimpleTight',
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
