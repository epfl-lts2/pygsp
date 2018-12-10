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
    Filter.approximate
    Filter.plot
    Filter.localize

Filters
-------

Then, derived classes implement various common graph filters.

**Low-pass filters**

.. autosummary::

    Heat

**Band-pass filters**

These filters can be configured to be low-pass, high-pass, or band-pass.

.. autosummary::

    Expwin
    Rectangular

**Filter banks of two filters: a low-pass and a high-pass**

.. autosummary::

    Regular
    Held
    Simoncelli
    Papadakis

**Filter banks composed of dilated or translated filters**

.. autosummary::

    Abspline
    HalfCosine
    Itersine
    MexicanHat
    Meyer
    SimpleTight

**Filter banks for vertex-frequency analyzes**

Those filter banks are composed of shifted versions of a mother filter, one per
graph frequency (Laplacian eigenvalue). They can analyze frequency content
locally, as a windowed graph Fourier transform.

.. autosummary::

    Gabor
    Modulation

Approximations
--------------

Moreover, two approximation methods are provided for fast filtering. The
computational complexity of filtering with those approximations is linear with
the number of edges. The complexity of the exact solution, which is to use the
Fourier basis, is quadratic with the number of nodes (without taking into
account the cost of the necessary eigendecomposition of the graph Laplacian).

**Chebyshev polynomials**

.. autosummary::

    Chebyshev

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
    'Modulation',
    'Papadakis',
    'Rectangular',
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

__all__ = _FILTERS + _APPROXIMATIONS + ['Chebyshev']

_utils.import_classes(_FILTERS, 'filters', 'filters')
_utils.import_functions(_APPROXIMATIONS, 'filters.approximations_old', 'filters')

from .approximations import *
