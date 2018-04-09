# -*- coding: utf-8 -*-

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class Gabor(Filter):
    r"""Design a Gabor filter bank.

    Design a filter bank where the kernel is centered at each graph frequency.

    Parameters
    ----------
    G : graph
    kernel : function
        Kernel function to be centered and evaluated.

    Examples
    --------
    >>> G = graphs.Logo()
    >>> k = lambda x: x / (1. - x)
    >>> g = filters.Gabor(G, k);

    """

    def __init__(self, G, kernel):

        Nf = G.e.shape[0]

        kernels = []
        for i in range(Nf):
            kernels.append(lambda x, i=i: kernel(x - G.e[i]))

        super(Gabor, self).__init__(G, kernels)
