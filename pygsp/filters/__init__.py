# -*- coding: utf-8 -*-

r"""
This module implements the main filter class and all the filters subclasses

:class: `Filter` Main filter class
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
    a : int
        Description.
    b : array_like
        Description.
    c : bool
        Description.

    Returns
    -------
    d : ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.filters.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)


__all__ = ['Filter', 'Abspline', 'Expwin', 'Gabor', 'HalfCosine', 'Heat', 'Held', 'Itersine', 'MexicanHat', 'Meyer', 'Papadakis', 'Regular', 'Simoncelli', 'SimpleTf', 'WarpedTranslates']


# Automaticaly import all classes from subfiles defined in __all__
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.filters'), class_to_import))
