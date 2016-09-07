# -*- coding: utf-8 -*-

r"""This module implements some PointsClouds."""

import importlib
import sys

__all__ = ['PointsCloud']


# Automaticaly import all classes from subfiles defined in __all__
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.pointsclouds'), class_to_import))
