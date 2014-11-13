# -*- coding: utf-8 -*-

r"""
Module documentation.
"""

import numpy as np


class Graph(object):

    def __init__(self, W, A, N, d, Ne, gtype, directed, lap_type,
                 L, k=12, **kwargs):
        self.M = N
        pass

    def copy_graph_attr(self, gtype, Gn):
        pass

    def separate_graph(self):
        pass

    def subgraph(self, c):
        pass


# Need M
class Grid2d(Graph):

    def __init__(self, M, **kwargs):
        super(Grid2d, self).__init__(**kwargs)
        pass


class Torus(Graph):

    def __init__(self, M, **kwargs):
        super(Torus, self).__init__(**kwargs)
        pass


# Need K
class Comet(Graph):

    def __init__(self, k, **kwargs):
        super(Comet, self).__init__(**kwargs)
        pass


class LowStretch(Graph):

    def __init__(self, k, **kwargs):
        super(LowStretch, self).__init__(**kwargs)
        pass


class RadomRegular(Graph):

    def __init__(self, k, **kwargs):
        super(RadomRegular, self).__init__(**kwargs)
        pass


class Ring(Graph):

    def __init__(self, k, **kwargs):
        super(Ring, self).__init__(**kwargs)
        pass


# Need params
class Community(Graph):

    def __init__(self, **kwargs):
        super(Community, self).__init__(**kwargs)
        pass


class Cube(Graph):

    def __init__(self, **kwargs):
        super(Cube, self).__init__(**kwargs)
        pass


class Sensor(Graph):

    def __init__(self, **kwargs):
        super(Sensor, self).__init__(**kwargs)
        pass


class Sphere(Graph):

    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        pass


# Need nothing
class Airfoil(Graph):

    def __init__(self):
        super(Airfoil, self).__init__()
        pass


class Bunny(Graph):

    def __init__(self):
        super(Bunny, self).__init__()
        pass


class DavidSensorNet(Graph):

    def __init__(self):
        super(DavidSensorNet, self).__init__()
        pass


class Full_connected(Graph):

    def __init__(self):
        super(Full_connected, self).__init__()
        pass


class Logo(Graph):

    def __init__(self):
        super(Logo, self).__init__()
        pass


class Path(Graph):

    def __init__(self):
        super(Path, self).__init__()
        pass


class RandomRing(Graph):

    def __init__(self):
        super(RandomRing, self).__init__()
        pass


def dummy(a, b, c):
    r"""
    Short description.

    Long description.

    Parameters
    ----------
    a : int
        Description.
    b : array_like
        Description.
    c : bool
        Description.

    Returns
    -------
    d : ndarray
        Description.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.module1.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    return np.array(b)
