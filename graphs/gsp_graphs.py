from numpy import array
from scipy.sparse import lil_matrix


class Graph:
    """ Graph class to define a single graph from a weight matrix or by default.
    coords and limits for the plotting can be specified too as arguments."""

    def __init__(self, W, *args):
        if W is None:
            # default matrix
            self.W = lil_matrix(0)
            self.graphtype = 'default'
        else:
            self.graphtype = 'from weight'

        # TODO add weight check

        narg = len(args)
        if narg >= 1:
            self.coords = args[0]
        if narg >= 2:
            self.limits = args[1]

        # Sparsing the matrix
        self.W = lil_matrix(W)
        self.A = lil_matrix(W > 0)
        self.N = W.size
