# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp` package is mainly organized around the following two modules:

* :mod:`.graphs` to create and manipulate various kinds of graphs,
* :mod:`.filters` to create and manipulate various graph filters.

Moreover, the following modules provide additional functionality:

* :mod:`.plotting` to plot,
* :mod:`.reduction` to reduce a graph while keeping its structure,
* :mod:`.features` to compute features on graphs,
* :mod:`.optimization` to help solving convex optimization problems,
* :mod:`.utils` for various utilities.

"""

from pygsp import utils as _utils

__all__ = [
    'graphs',
    'filters',
    'plotting',
    'reduction',
    'features',
    'optimization',
    'utils',
]

_utils.import_modules(__all__[::-1], 'pygsp', 'pygsp')

__version__ = '0.5.1'
__release_date__ = '2017-12-15'
