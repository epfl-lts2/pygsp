# -*- coding: utf-8 -*-

r"""
This module implements graphs

:class: `Graph`  Main graph class
"""

import numpy as np
import importlib
import sys


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a: int
        Description.
    b: array_like
        Description.
    c: bool
        Description.

    Returns
    -------
    d: ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.graphs.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)


__all__ = ['Graph', 'Airfoil', 'Comet', 'Community', 'DavidSensorNet', 'FullConnected', 'Grid2d', 'Logo', 'LowStretchTree', 'Minnesota', 'Path', 'RandomRing', 'Ring', 'Sensor', 'SwissRoll', 'Torus']


# Automaticaly import all classes from subfiles defined in __all__
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.graphs'), class_to_import))
from nngraphs import *
