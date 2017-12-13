# -*- coding: utf-8 -*-

from __future__ import division

import collections
import copy

import numpy as np
from scipy import sparse, spatial

from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5


class Community(Graph):
    r"""Community graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 256).
    Nc : int (optional)
        Number of communities (default = :math:`\lfloor \sqrt{N}/2 \rceil`).
    min_comm : int (optional)
        Minimum size of the communities
        (default = :math:`\lfloor N/Nc/3 \rceil`).
    min_deg : int (optional)
        NOT IMPLEMENTED. Minimum degree of each node (default = 0).
    comm_sizes : int (optional)
        Size of the communities (default = random).
    size_ratio : float (optional)
        Ratio between the radius of world and the radius of communities
        (default = 1).
    world_density : float (optional)
        Probability of a random edge between two different communities
        (default = 1/N).
    comm_density : float (optional)
        Probability of a random edge inside any community (default = None,
        which implies k_neigh or epsilon will be used to determine
        intra-edges).
    k_neigh : int (optional)
        Number of intra-community connections.
        Not used if comm_density is defined (default = None, which implies
        comm_density or epsilon will be used to determine intra-edges).
    epsilon : float (optional)
        Largest distance at which two nodes sharing a community are connected.
        Not used if k_neigh or comm_density is defined
        (default = :math:`\sqrt{2\sqrt{N}}/2`).
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Community(N=250, Nc=3, comm_sizes=[50, 120, 80], seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.5)
    >>> G.plot(ax=axes[1])

    """
    def __init__(self,
                 N=256,
                 Nc=None,
                 min_comm=None,
                 min_deg=0,
                 comm_sizes=None,
                 size_ratio=1,
                 world_density=None,
                 comm_density=None,
                 k_neigh=None,
                 epsilon=None,
                 seed=None,
                 **kwargs):

        if Nc is None:
            Nc = int(round(np.sqrt(N) / 2))
        if min_comm is None:
            min_comm = int(round(N / (3 * Nc)))
        if world_density is None:
            world_density = 1 / N
        if not 0 <= world_density <= 1:
            raise ValueError('World density should be in [0, 1].')
        if epsilon is None:
            epsilon = np.sqrt(2 * np.sqrt(N)) / 2
        rs = np.random.RandomState(seed)

        self.logger = utils.build_logger(__name__)
        w_data = [[], [[], []]]

        if min_comm * Nc > N:
            raise ValueError('The constraint on minimum size for communities is unsolvable.')

        info = {'node_com': None, 'comm_sizes': None, 'world_rad': None,
                'world_density': world_density, 'min_comm': min_comm}

        # Communities construction #
        if comm_sizes is None:
            mandatory_labels = np.tile(np.arange(Nc), (min_comm,))  # min_comm labels for each of the Nc communities
            remaining_labels = rs.choice(Nc, N - min_comm * Nc)  # random choice for the remaining labels
            info['node_com'] = np.sort(np.concatenate((mandatory_labels, remaining_labels)))
        else:
            if len(comm_sizes) != Nc:
                raise ValueError('There should be Nc community sizes.')
            if np.sum(comm_sizes) != N:
                raise ValueError('The sum of community sizes should be N.')
            # create labels based on the constraint given for the community sizes. No random assignation here.
            info['node_com'] = np.concatenate([[val] * cnt for (val, cnt) in enumerate(comm_sizes)])

        counts = collections.Counter(info['node_com'])
        info['comm_sizes'] = np.array([cnt[1] for cnt in sorted(counts.items())])
        info['world_rad'] = size_ratio * np.sqrt(N)

        # Intra-community edges construction #
        if comm_density is not None:
            # random picking edges following the community density (same for all communities)
            comm_density = float(comm_density)
            comm_density = comm_density if 0. <= comm_density <= 1. else 0.1
            info['comm_density'] = comm_density
            self.logger.info('Constructed using community density = {}'.format(comm_density))
        elif k_neigh is not None:
            # k-NN among the nodes in the same community (same k for all communities)
            if k_neigh < 0:
                raise ValueError('k_neigh cannot be negative.')
            info['k_neigh'] = k_neigh
            self.logger.info('Constructed using K-NN with k = {}'.format(k_neigh))
        else:
            # epsilon-NN among the nodes in the same community (same eps for all communities)
            info['epsilon'] = epsilon
            self.logger.info('Constructed using eps-NN with eps = {}'.format(epsilon))

        # Coordinates #
        info['com_coords'] = info['world_rad'] * np.array(list(zip(
            np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc),
            np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))

        coords = rs.rand(N, 2)  # nodes' coordinates inside the community
        coords = np.array([[elem[0] * np.cos(2 * np.pi * elem[1]),
                            elem[0] * np.sin(2 * np.pi * elem[1])] for elem in coords])

        for i in range(N):
            # set coordinates as an offset from the center of the community it belongs to
            comm_idx = info['node_com'][i]
            comm_rad = np.sqrt(info['comm_sizes'][comm_idx])
            coords[i] = info['com_coords'][comm_idx] + comm_rad * coords[i]

        first_node = 0
        for i in range(Nc):
            com_siz = info['comm_sizes'][i]
            M = com_siz * (com_siz - 1) / 2

            if comm_density is not None:
                nb_edges = int(comm_density * M)
                tril_ind = np.tril_indices(com_siz, -1)
                indices = rs.permutation(int(M))[:nb_edges]

                w_data[0] += [1] * nb_edges
                w_data[1][0] += [first_node + tril_ind[1][elem] for elem in indices]
                w_data[1][1] += [first_node + tril_ind[0][elem] for elem in indices]

            elif k_neigh is not None:
                comm_coords = coords[first_node:first_node + com_siz]
                kdtree = spatial.KDTree(comm_coords)
                __, indices = kdtree.query(comm_coords, k=k_neigh + 1)

                pairs_set = set()
                map(lambda row: map(lambda elm: pairs_set.add((min(row[0], elm), max(row[0], elm))), row[1:]), indices)

                w_data[0] += [1] * len(pairs_set)
                w_data[1][0] += [first_node + pair[0] for pair in pairs_set]
                w_data[1][1] += [first_node + pair[1] for pair in pairs_set]

            else:
                comm_coords = coords[first_node:first_node + com_siz]
                kdtree = spatial.KDTree(comm_coords)
                pairs_set = kdtree.query_pairs(epsilon)

                w_data[0] += [1] * len(pairs_set)
                w_data[1][0] += [first_node + elem[0] for elem in pairs_set]
                w_data[1][1] += [first_node + elem[1] for elem in pairs_set]

            first_node += com_siz

        # Inter-community edges construction #
        M = (N**2 - np.sum([com_siz**2 for com_siz in info['comm_sizes']])) / 2
        nb_edges = int(world_density * M)

        if world_density < 0.35:
            # use regression sampling
            inter_edges = set()
            while len(inter_edges) < nb_edges:
                new_point = rs.randint(0, N, 2)
                if info['node_com'][min(new_point)] != info['node_com'][max(new_point)]:
                    inter_edges.add((min(new_point), max(new_point)))
        else:
            # use random permutation
            indices = rs.permutation(int(M))[:nb_edges]
            all_points, first_col = [], 0
            for i in range(Nc - 1):
                nb_col = info['comm_sizes'][i]
                first_row = np.sum(info['comm_sizes'][:i+1])

                for j in range(i+1, Nc):
                    nb_row = info['comm_sizes'][j]
                    all_points += [(first_row + r, first_col + c) for r in range(nb_row) for c in range(nb_col)]

                    first_row += nb_row
                first_col += nb_col

            inter_edges = np.array(all_points)[indices]

        w_data[0] += [1] * nb_edges
        w_data[1][0] += [elem[0] for elem in inter_edges]
        w_data[1][1] += [elem[1] for elem in inter_edges]

        w_data[0] += w_data[0]
        tmp_w_data = copy.deepcopy(w_data[1][0])
        w_data[1][0] += w_data[1][1]
        w_data[1][1] += tmp_w_data
        w_data[1] = tuple(w_data[1])

        W = sparse.coo_matrix(tuple(w_data), shape=(N, N))

        for key, value in {'Nc': Nc, 'info': info}.items():
            setattr(self, key, value)

        super(Community, self).__init__(W=W, gtype='Community', coords=coords, **kwargs)
