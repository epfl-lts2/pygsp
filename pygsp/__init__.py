# -*- coding: utf-8 -*-

"""
Package documentation.
"""

# When importing the toolbox, you surely want these modules.
from pygsp import utils
from pygsp import graphs
try:
    from pygsp import plotting
except:
    pass
from pygsp import operators
from pygsp import filters

# Silence the code checker warning about unused symbols.
assert utils
assert graphs
try:
    assert plotting
except:
    pass
assert operators
assert filters

__version__ = '0.0.1'
__release_date__ = '2014-10-06'
