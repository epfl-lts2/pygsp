# -*- coding: utf-8 -*-

r"""
This module implements filters and contains predefined filters that can be directly applied to graphs.

A filter is associated to a graph and is defined with one or several function(s).
We define by Filterbank a list of filters applied to a single graph.
Tools for the analysis, the synthesis and the evaluation are provided to work with the filters on the graphs.
For specific information, :ref:`see details  here<graphs-api>`.
"""

import importlib
import sys


__all__ = ['Filter', 'Abspline', 'Expwin', 'Gabor', 'HalfCosine', 'Heat', 'Held', 'Itersine', 'MexicanHat', 'Meyer',
           'Papadakis', 'Regular', 'Simoncelli', 'SimpleTf', 'WarpedTranslates']


# Automaticaly import all classes from subfiles defined in __all__
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.filters'), class_to_import))
