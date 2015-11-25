# -*- coding: utf-8 -*-

import numpy as np
from scipy import io

from os import path


class PointsCloud(object):
    r"""
    Load the parameters of models and the points.

    Parameters
    ----------
    name : string
        The name of the point cloud to load.
        Possible arguments : 'airfoil', 'bunny', 'david64', 'david500', 'logo',
        'minnesota', two_moons'.
    max_dim : int
        The maximum dimensionality of the points (only valid for two_moons)
        (default is 2)

    Returns
    -------
    The differents informations of the loaded PointsCloud.


    Examples
    --------
    >>> from pygsp import pointsclouds
    >>> bunny = pointsclouds.PointsCloud('bunny')
    >>> Xin = bunny.Xin


    Note
    ----
    The bunny is the model from the Stanford Computer Graphics Laboratory
    (see reference).


    References
    ----------
    See :cite:`turk1994zippered` for more informations.

    """

    def __init__(self, pointcloudname, max_dim=2):
        if pointcloudname == "airfoil":
            airfoilmat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'airfoil.mat'))
            self.i_inds = airfoilmat['i_inds']
            self.j_inds = airfoilmat['j_inds']
            self.x = airfoilmat['x']
            self.y = airfoilmat['y']
            self.coords = np.concatenate((self.x, self.y), axis=1)

        elif pointcloudname == "bunny":
            bunnymat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'bunny.mat'))
            self.Xin = bunnymat["bunny"]

        elif pointcloudname == "david64":
            david64mat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'david64.mat'))
            self.W = david64mat["W"]
            self.N = david64mat["N"][0, 0]
            self.coords = david64mat["coords"]

        elif pointcloudname == "david500":
            david500mat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'david500.mat'))
            self.W = david500mat["W"]
            self.N = david500mat["N"][0, 0]
            self.coords = david500mat["coords"]

        elif pointcloudname == "logo":
            logomat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'logogsp.mat'))
            self.W = logomat["W"]
            self.coords = logomat["coords"]
            self.limits = np.array([0, 640, -400, 0])

            self.info = {"idx_g": logomat["idx_g"],
                         "idx_s": logomat["idx_s"],
                         "idx_p": logomat["idx_p"]}

        elif pointcloudname == "minnesota":
            minnesotamat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'minnesota.mat'))
            self.A = minnesotamat["A"]
            self.labels = minnesotamat["labels"]
            self.coords = minnesotamat["xy"]

        elif pointcloudname == "two_moons":
            twomoonsmat = io.loadmat(path.join(path.dirname(
                path.realpath(__file__)), 'misc', 'two_moons.mat'))
            if max_dim == -1:
                max_dim == 2
            self.Xin = twomoonsmat["features"][:max_dim].T

        else:
            raise ValueError("This PointsCloud does not exist. Please verify "
                             "you wrote the right name in lower case.")

    def plot(self, **kwargs):
        r"""
        Plot the pointcloud.

        See plotting doc.
        """
        from pygsp import plotting
        plotting.plot_pointcloud(self, **kwargs)
