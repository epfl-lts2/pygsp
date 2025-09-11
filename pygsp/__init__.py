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

from . import features  # noqa: F401
from . import filters  # noqa: F401
from . import graphs  # noqa: F401
from . import learning  # noqa: F401
from . import optimization  # noqa: F401
from . import plotting  # noqa: F401
from . import reduction  # noqa: F401
from . import utils  # noqa: F401

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

__version__ = "0.6.1"
__release_date__ = "2025-09-11"
