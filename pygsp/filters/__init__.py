# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.filters` module implements methods used for filtering (e.g.
analysis, synthesis, evaluation) and defines commonly used filters that can be
applied to :mod:`pygsp.graphs`. A filter is associated to a graph and is
defined with one or several functions. We define by filterbank a list of
filters, usually centered around different frequencies, applied to a single
graph.

The :class:`Filter` base class implements a common interface to all filters:

* :meth:`Filter.evaluate`: evaluate frequency response of the filterbank
* :meth:`Filter.analysis`: compute signal response to the filterbank
* :meth:`Filter.synthesis`: synthesize signal from response
* :meth:`Filter.compute_frame`: return a matrix operator
* :meth:`Filter.estimate_frame_bounds`: estimate lower and upper frame bounds
* :meth:`Filter.plot`: plot the filter frequency response
* :meth:`Filter.localize`: localize a kernel at a node (to visualize it)

Then, derived classes implement various common graph filters.

**Filter banks of N filters**

* :class:`Abspline`: design a absspline filter bank
* :class:`Gabor`: design a Gabor filter bank
* :class:`HalfCosine`: design a half cosine filter bank (tight frame)
* :class:`Itersine`: design an itersine filter bank (tight frame)
* :class:`MexicanHat`: design a mexican hat filter bank
* :class:`Meyer`: design a Meyer filter bank
* :class:`SimpleTight`: design a simple tight frame filter bank (tight frame)

**Filter banks of 2 filters: a low pass and a high pass**

* :class:`Regular`: design 2 filters with the regular construction
* :class:`Held`: design 2 filters with the Held construction
* :class:`Simoncelli`: design 2 filters with the Simoncelli construction
* :class:`Papadakis`: design 2 filters with the Papadakis construction

**Low pass filters**

* :class:`Heat`: design an heat kernel filter
* :class:`Expwin`: design an exponential window filter


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
