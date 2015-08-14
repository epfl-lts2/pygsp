# -*- coding: utf-8 -*-

"""
Package documentation.
"""

# When importing the toolbox, you surely want these modules.
from pygsp import graphs
from pygsp import operators
from pygsp import utils
from pygsp import filters
from pygsp import pointsclouds
from pygsp import data_handling
from pygsp import optimization
from pygsp import reduction

# Silence the code checker warning about unused symbols.
assert graphs
assert utils
assert filters
assert operators
assert pointsclouds
assert data_handling
assert optimization
assert reduction

__version__ = '0.0.1'
__email__ = 'LTS2Graph@groupes.epfl.ch'
__release_date__ = '2014-10-06'
