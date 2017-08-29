# -*- coding: utf-8 -*-

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class Gabor(Filter):
    r"""Design a Gabor filter bank.

    Design a filter bank where the kernel *k* is placed at each graph
    frequency.

    Parameters
    ----------
    G : graph
    k : lambda function
        kernel

    Examples
    --------
    >>> G = graphs.Logo()
    >>> k = lambda x: x / (1. - x)
    >>> g = filters.Gabor(G, k);

    """
    def __init__(self, G, k):

        Nf = G.e.shape[0]

        g = []
        for i in range(Nf):
            g.append(lambda x, ii=i: k(x - G.e[ii]))

        super(Gabor, self).__init__(G, g)
