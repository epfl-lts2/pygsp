# -*- coding: utf-8 -*-

r"""
This module implements graphs and contains predefined graphs for the most famous ones.

A graph is constructed either from its adjacency matrix, its weight matrix or any other parameter
which depends on the particular graph you are trying to build. For specific information, :ref:`see details  here<graphs-api>`.
"""

import importlib
import sys

__all__ = ['Graph', 'Airfoil', 'BarabasiAlbert', 'Comet', 'Community', 'DavidSensorNet', 'ErdosRenyi', 'FullConnected', 'Grid2d', 'Logo',
           'LowStretchTree', 'Minnesota', 'Path', 'RandomRing', 'RandomRegular', 'Ring', 'Sensor', 'StochasticBlockModel', 'SwissRoll', 'Torus']



# Automaticaly import all classes from subfiles defined in __all__
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.graphs'), class_to_import))

from .nngraphs import *
from .gutils import *
