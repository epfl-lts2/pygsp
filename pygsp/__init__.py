# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp` package is mainly organized around the following three
modules:

* :mod:`.graphs` to create and manipulate various kinds of graphs,
* :mod:`.filters` to create and manipulate various graph filters,
* :mod:`.operators` to apply various operators to graph signals.

Moreover, the following modules provide additional functionality:

* :mod:`.plotting` to plot,
* :mod:`.features` to compute features on graphs,
* :mod:`.optimization` to help solving convex optimization problems,
* :mod:`.utils` for various utilities.

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
