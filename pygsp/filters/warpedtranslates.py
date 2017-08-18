# -*- coding: utf-8 -*-

from . import Filter


class WarpedTranslates(Filter):
    r"""
    Vertex frequency filterbank

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters

    References
    ----------
    See :cite:`shuman2013spectrum`

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.WarpedTranslates(G)
    Traceback (most recent call last):
      ...
    NotImplementedError

    """

    def __init__(self, G, Nf=6, **kwargs):
        raise NotImplementedError
