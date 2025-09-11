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
    Filter.complement
    Filter.inverse
    Filter.compute_frame
    Filter.estimate_frame_bounds
    Filter.plot
    Filter.localize

Filters
-------

Then, derived classes implement various common graph filters.

**Filters that solve differential equations**

The following filters solve partial differential equations (PDEs) on graphs,
which model processes such as heat diffusion or wave propagation.

.. autosummary::

    Heat
    Wave

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

    compute_cheby_coeff
    compute_jackson_cheby_coeff
    cheby_op
    cheby_rect

**Lanczos algorithm**

.. autosummary::

    lanczos
    lanczos_op

"""

from .abspline import Abspline  # noqa: F401
from .approximations import cheby_op  # noqa: F401
from .approximations import cheby_rect  # noqa: F401
from .approximations import compute_cheby_coeff  # noqa: F401
from .approximations import compute_jackson_cheby_coeff  # noqa: F401
from .approximations import lanczos  # noqa: F401
from .approximations import lanczos_op  # noqa: F401
from .expwin import Expwin  # noqa: F401
from .filter import Filter  # noqa: F401
from .gabor import Gabor  # noqa: F401
from .halfcosine import HalfCosine  # noqa: F401
from .heat import Heat  # noqa: F401
from .held import Held  # noqa: F401
from .itersine import Itersine  # noqa: F401
from .mexicanhat import MexicanHat  # noqa: F401
from .meyer import Meyer  # noqa: F401
from .modulation import Modulation  # noqa: F401
from .papadakis import Papadakis  # noqa: F401
from .rectangular import Rectangular  # noqa: F401
from .regular import Regular  # noqa: F401
from .simoncelli import Simoncelli  # noqa: F401
from .simpletight import SimpleTight  # noqa: F401
from .wave import Wave  # noqa: F401
