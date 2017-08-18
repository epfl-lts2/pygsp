# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg

from . import Filter
from pygsp import utils


_logger = utils.build_logger(__name__)


class Heat(Filter):
    r"""
    Heat filterbank

    Parameters
    ----------
    G : graph
    tau : int or list of ints
        Scaling parameter. (default = 10)
    normalize : bool
        Normalize the kernel (works only if the eigenvalues are
        present in the graph). (default = 0)

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Heat(G)

    """

    def __init__(self, G, tau=10, normalize=False, **kwargs):

        g = []

        if normalize:
            if not hasattr(G, 'e'):
                _logger.info('Filter Heat will calculate and set '
                             'the eigenvalues to normalize the kernel')
                G.compute_fourier_basis()

            if isinstance(tau, list):
                for t in tau:
                    def gu(x, taulam=t):
                        return np.exp(-taulam * x/G.lmax)
                    ng = linalg.norm(gu(G.e))
                    g.append(lambda x, taulam=t: np.exp(-taulam *
                                                        x/G.lmax / ng))
            else:
                def gu(x):
                    return np.exp(-tau * x/G.lmax)
                ng = linalg.norm(gu(G.e))
                g.append(lambda x: np.exp(-tau * x/G.lmax / ng))

        else:
            if isinstance(tau, list):
                for t in tau:
                    g.append(lambda x, taulam=t: np.exp(-taulam * x/G.lmax))
            else:
                g.append(lambda x: np.exp(-tau * x/G.lmax))

        super(Heat, self).__init__(G, g, **kwargs)
