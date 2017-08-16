# -*- coding: utf-8 -*-

"""
This toolbox is splitted in different modules taking care of the different
aspects of Graph Signal Processing.

Those modules are : :ref:`Graphs <graphs-api>`, :ref:`Filters <filters-api>`,
:ref:`Operators <operators-api>`, :ref:`PointCloud <pointclouds-api>`,
:ref:`Plotting <plotting-api>`, :ref:`Data Handling <data_handling-api>` and
:ref:`Utils <utils-api>`.

You can find detailed documentation on the use of the functions in the
subsequent pages.
"""

from pygsp import utils as _utils

__all__ = [
    'graphs',
    'filters',
    'operators',
    'plotting',
    'features',
    'data_handling',
    'optimization',
    'utils',
]

_utils.import_modules(__all__[::-1], 'pygsp', 'pygsp')

__version__ = '0.4.2'
__release_date__ = '2017-04-27'
