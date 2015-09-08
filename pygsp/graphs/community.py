# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import distanz

import numpy as np
from scipy import sparse, spatial
import random as rd
from math import sqrt


class Community(Graph):
    r"""
    Create a community graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 256)
    Nc : int
        Number of communities (default = round(sqrt(N)/2))
    com_sizes : int
        Size of the communities (default = random)
    min_comm : int
        Minimum size of the communities (default = round(N/Nc/3))
    min_deg : int
        Minimum degree of each node (default = round(min_comm/2)
        (not implemented yet))
    size_ratio : float
        Ratio between the radius of world and the radius of communities
        (default = 1)
    world_density : float
        Probability of a random edge between any pair of edges (default = 1/N)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Community()

    """

    def __init__(self, N=256, Nc=None, com_sizes=np.array([]), min_com=None,
                 min_deg=None, size_ratio=1, world_density=None, **kwargs):
        # Initialisation of the parameters
        if not Nc:
            Nc = int(round(sqrt(N)/2.))

        if len(com_sizes) != 0:
            if np.sum(com_sizes) != N:
                raise ValueError("GSP_COMMUNITY: The sum of the community \
                                 sizes has to be equal to N")

        if not min_com:
            min_com = int(round(float(N) / Nc / 3.))

        if not min_deg:
            min_deg = round(min_com/2.)

        if world_density is None:
            world_density = 1./N

        # Begining
        if np.shape(com_sizes)[0] == 0:
            x = N - (min_com - 1)*Nc - 1
            com_lims = np.sort(np.resize(np.random.permutation(int(x)), (Nc - 1.))) + 1
            com_lims += np.cumsum((min_com-1)*np.ones(np.shape(com_lims)))
            com_lims = np.concatenate((np.array([0]), com_lims, np.array([N])))
            com_sizes = np.diff(com_lims)

        if False:  # Verbose > 2 ?
            X = np.zeros((10000, Nc + 1))
            # pick randomly param.Nc-1 points to cut the rows in communtities:
            for i in range(10000):
                com_lims_tmp = np.sort(np.resize(np.random.permutation(int(x)),
                                                 (Nc - 1.))) + 1
                com_lims_tmp += np.cumsum((min_com - 1) *
                                          np.ones(np.shape(com_lims_tmp)))
                X[i, :] = np.concatenate((np.array([0]), com_lims_tmp,
                                          np.array([N])))
                # dX = np.diff(X.T).T

            for i in range(int(Nc)):
                # TODO figure; hist(dX(:,i), 100); title('histogram of
                # row community size'); end
                pass
            del X
            del com_lims_tmp

        rad_world = size_ratio*sqrt(N)
        com_coords = rad_world*np.concatenate((
            -np.expand_dims(np.cos(2*np.pi*(np.arange(Nc) + 1)/Nc), axis=1),
            np.expand_dims(np.sin(2*np.pi*(np.arange(Nc) + 1)/Nc), axis=1)),
            axis=1)

        coords = np.ones((N, 2))

        # create uniformly random points in the unit disc
        for i in range(N):
            # use rejection sampling to sample from a unit disc
            # (probability = pi/4)
            while np.linalg.norm(coords[i], 2) >= 0.5:
                # sample from the square and reject anything outside the circle
                coords[i] = rd.random() - 0.5, rd.random() - 0.5

        info = {"node_com": np.zeros((N, 1))}

        # add the offset for each node depending on which community
        # it belongs to
        for i in range(int(Nc)):
            com_size = com_sizes[i]
            rad_com = sqrt(com_size)

            node_ind = np.arange(com_lims[i], com_lims[i + 1])
            coords[node_ind] = rad_com*coords[node_ind] + com_coords[i]
            info["node_com"][node_ind] = i

        # Look for epsilon-niehgbors of the nodes, with distance < 1e-3
        kdtree = spatial.KDTree(coords)
        idx_w = kdtree.query_pairs(np.sqrt(N/4./Nc))

        # Add noise with approximative world_density: no double edges, no self loops
        for (i_idx, j_idx) in np.random.randint(0, high=N, size=(int(N*N*world_density/2), 2)):
            if i_idx != j_idx:
                idx_w.add((i_idx, j_idx) if i_idx < j_idx else (j_idx, i_idx))

        if idx_w:
            idx_w = np.array(list(idx_w))  # transform into array
            idx_w = np.concatenate((idx_w, np.fliplr(idx_w)))
            self.W = sparse.coo_matrix((np.ones(idx_w.shape[0]), (idx_w[:, 0], idx_w[:, 1])), shape=(N, N))
        else:
            self.W = sparse.coo_matrix((N, N))
        self.gtype = "Community"
        self.coords = coords
        self.N = N
        self.Nc = Nc

        # return additional info about the communities
        info["com_lims"] = com_lims
        info["com_coords"] = com_coords
        info["com_sizes"] = com_sizes
        self.info = info

        super(Community, self).__init__(W=self.W, gtype=self.gtype,
                                        coords=self.coords, info=self.info, **kwargs)
