# -*- coding: utf-8 -*-

from . import Filter


class WarpedTranslates(Filter):
    r"""
    Creates a vertex frequency filterbank

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters

    Returns
    -------
    out : WarpedTranslates

    Examples
    --------
    Not Implemented for now
    # >>> from pygsp import graphs, filters
    # >>> G = graphs.Logo()
    # >>> F = filters.WarpedTranslates(G)

    See :cite:`shuman2013spectrum`

    """

    def __init__(self, G, Nf=6, **kwargs):
        super(WarpedTranslates, self).__init__(G, **kwargs)
        raise NotImplementedError
