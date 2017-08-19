# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp` package is mainly organized around the following three
modules:

* :mod:`pygsp.graphs` to create and manipulate various kinds of graphs,
* :mod:`pygsp.filters` to create and manipulate various graph filters,
* :mod:`pygsp.operators` to apply various operators to graph signals.

Moreover, the following modules provide additional functionality:

* :mod:`pygsp.plotting` to plot,
* :mod:`pygsp.features` to compute features on graphs,
* :mod:`pygsp.optimization` to help solving convex optimization problems,
* :mod:`pygsp.utils` for various utilities.

"""

from pygsp import utils as _utils

__all__ = [
    'graphs',
    'filters',
    'operators',
    'plotting',
    'features',
    'optimization',
    'utils',
]

_utils.import_modules(__all__[::-1], 'pygsp', 'pygsp')

__version__ = '0.4.2'
__release_date__ = '2017-04-27'
