# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp` package is mainly organized around the following two modules:

* :mod:`.graphs` to create and manipulate various kinds of graphs,
* :mod:`.filters` to create and manipulate various graph filters.

Moreover, the following modules provide additional functionality:

* :mod:`.plotting` to plot,
* :mod:`.reduction` to reduce a graph while keeping its structure,
* :mod:`.features` to compute features on graphs,
* :mod:`.learning` to solve learning problems,
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
    'learning',
    'optimization',
    'utils',
]

_utils.import_modules(__all__[::-1], 'pygsp', 'pygsp')

# Users only call the plot methods from the objects.
# It's thus more convenient for them to have the doc there.
# But it's more convenient for developers to have the doc alongside the code.
try:
    filters.Filter.plot.__doc__ = plotting._plot_filter.__doc__
    graphs.Graph.plot.__doc__ = plotting._plot_graph.__doc__
    graphs.Graph.plot_spectrogram.__doc__ = plotting._plot_spectrogram.__doc__
except AttributeError:
    # For Python 2.7.
    filters.Filter.plot.__func__.__doc__ = plotting._plot_filter.__doc__
    graphs.Graph.plot.__func__.__doc__ = plotting._plot_graph.__doc__
    graphs.Graph.plot_spectrogram.__func__.__doc__ = plotting._plot_spectrogram.__doc__

__version__ = '0.5.1'
__release_date__ = '2017-12-15'
