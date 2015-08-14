# -*- coding: utf-8 -*-

import importlib
import sys


__all__ = ['NNGraph', 'Bunny', 'Cube', 'Sphere', 'TwoMoons']
for class_to_import in __all__:
    setattr(sys.modules[__name__], class_to_import, getattr(importlib.import_module('.' + class_to_import.lower(), 'pygsp.graphs.nngraphs'), class_to_import))
