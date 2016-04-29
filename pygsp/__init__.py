# -*- coding: utf-8 -*-

"""
This toolbox is splitted in different modules taking care of the different aspects of Graph Signal Processing.

Those modules are : Graphs, Filters, Operators and PointClouds.

You can find detailed documentation on the use of the functions in the subsequent pages.
"""

# When importing the toolbox, you surely want these modules.
from pygsp import graphs
from pygsp import operators
from pygsp import utils
from pygsp import features
from pygsp import filters
from pygsp import pointsclouds
from pygsp import data_handling
from pygsp import optimization
from pygsp import plotting

# Silence the code checker warning about unused symbols.
assert data_handling
assert filters
assert graphs
assert operators
assert optimization
assert pointsclouds
assert plotting
assert utils

__version__ = '0.4.0'
__email__ = 'LTS2Graph@groupes.epfl.ch'
__release_date__ = '2015-11-27'
