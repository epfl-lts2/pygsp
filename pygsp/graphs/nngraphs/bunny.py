# -*- coding: utf-8 -*-

from pygsp import utils
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


class Bunny(NNGraph):
    r"""Stanford bunny (NN-graph).

    References
    ----------
    See :cite:`turk1994zippered`.

    Examples
    --------
    >>> G = graphs.Bunny()

    """

    def __init__(self, **kwargs):

        data = utils.loadmat('pointclouds/bunny')

        plotting = {'vertex_size': 10,
                    'elevation': -89,
                    'azimuth': 94,
                    'distance': 7}

        super(Bunny, self).__init__(Xin=data['bunny'], epsilon=0.2,
                                    NNtype='radius', plotting=plotting,
                                    gtype='Bunny', **kwargs)
