# -*- coding: utf-8 -*-

"""
Package documentation.
"""

# When importing the toolbox, you surely want these modules.
from pygsp import utils
from pygsp import graphs
from pygsp import plotting
from pygsp import operators
from pygsp import filters

# Silence the code checker warning about unused symbols.
assert utils
assert graphs
assert plotting
assert operators
assert filters

__version__ = '0.0.1'
__email__ = 'LTS2Graph@groupes.epfl.ch'
__release_date__ = '2014-10-06'
