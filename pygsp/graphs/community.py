# -*- coding: utf-8 -*-

from . import Graph
from pygsp.utils import build_logger

from collections import Counter
from copy import deepcopy
import numpy as np
from scipy import sparse, spatial


class Community(Graph):
    r"""
    Create a community graph.

    Parameters
    ----------
    N : int
        Number of nodes (default = 256)
    Nc : int (optional)
        Number of communities (default = :math:`round(\sqrt{N}/2)`)
    min_comm : int (optional)
        Minimum size of the communities (default = round(N/Nc/3))
    min_deg : int (optional)
        Minimum degree of each node (default = 0, NOT IMPLEMENTED YET)
    comm_sizes : int (optional)
        Size of the communities (default = random)
    size_ratio : float (optional)
        Ratio between the radius of world and the radius of communities (default = 1)
    world_density : float (optional)
        Probability of a random edge between two different communities (default = 1/N)
    comm_density : float (optional)
        Probability of a random edge inside any community (default = None, not used if None)
    k_neigh : int (optional)
        Number of intra-community connections (default = None, not used if None or comm_density is defined)
    epsilon : float (optional)
        Max distance at which two nodes sharing a community are connected
        (default = :math:`sqrt(2\sqrt{N})/2`, not used if k_neigh or comm_density is defined)

    Examples
    --------
    >>> from pygsp import graphs
    >>> G = graphs.Community()

    """
    def __init__(self, N=256, **kwargs):

        # Parameter initialisation #
        N = int(N)
        Nc = int(kwargs.pop('Nc', int(round(np.sqrt(N)/2.))))
        min_comm = int(kwargs.pop('min_comm', int(round(N / (3. * Nc)))))
        min_deg = int(kwargs.pop('min_deg', 0))
        comm_sizes = kwargs.pop('comm_sizes', np.array([]))
        size_ratio = float(kwargs.pop('size_ratio', 1.))
        world_density = float(kwargs.pop('world_density', 1. / N))
        world_density = world_density if 0 <= world_density <= 1 else 1. / N
        comm_density = kwargs.pop('comm_density', None)
        k_neigh = kwargs.pop('k_neigh', None)
        epsilon = float(kwargs.pop('epsilon', np.sqrt(2 * np.sqrt(N)) / 2))

        self.logger = build_logger(__name__, **kwargs)
        w_data = [[], [[], []]]

        try:
            if len(comm_sizes) > 0:
                if np.sum(comm_sizes) != N:
                    raise ValueError('GSP_COMMUNITY: The sum of the community sizes has to be equal to N.')
                if len(comm_sizes) != Nc:
                    raise ValueError('GSP_COMMUNITY: The length of the community sizes has to be equal to Nc.')

        except TypeError:
            raise TypeError("GSP_COMMUNITY: comm_sizes expected to be a list or array, got {}".format(type(comm_sizes)))

        if min_comm * Nc > N:
            raise ValueError('GSP_COMMUNITY: The constraint on minimum size for communities is unsolvable.')

        info = {'node_com': None, 'comm_sizes': None, 'world_rad': None,
                'world_density': world_density, 'min_comm': min_comm}

        # Communities construction #
        if comm_sizes.shape[0] == 0:
            mandatory_labels = np.tile(np.arange(Nc), (min_comm,))  # min_comm labels for each of the Nc communities
            remaining_labels = np.random.choice(Nc, N - min_comm * Nc)  # random choice for the remaining labels
            info['node_com'] = np.sort(np.concatenate((mandatory_labels, remaining_labels)))
        else:
            # create labels based on the constraint given for the community sizes. No random assignation here.
            info['node_com'] = np.concatenate([[val] * cnt for (val, cnt) in enumerate(comm_sizes)])

        counts = Counter(info['node_com'])
        info['comm_sizes'] = np.array([cnt[1] for cnt in sorted(counts.items())])
        info['world_rad'] = size_ratio * np.sqrt(N)

        # Intra-community edges construction #
        if comm_density:
            # random picking edges following the community density (same for all communities)
            comm_density = float(comm_density)
            comm_density = comm_density if 0. <= comm_density <= 1. else 0.1
            info['comm_density'] = comm_density
            self.logger.info("GSP_COMMUNITY: Constructed using community density = {}".format(comm_density))
        elif k_neigh:
            # k-NN among the nodes in the same community (same k for all communities)
            k_neigh = int(k_neigh)
            k_neigh = k_neigh if k_neigh > 0 else 10
            info['k_neigh'] = k_neigh
            self.logger.info("GSP_COMMUNITY: Constructed using K-NN with k = {}".format(k_neigh))
        else:
            # epsilon-NN among the nodes in the same community (same eps for all communities)
            info['epsilon'] = epsilon
            self.logger.info("GSP_COMMUNITY: Constructed using eps-NN with eps = {}".format(epsilon))

        # Coordinates #
        info['com_coords'] = info['world_rad'] * np.array(list(zip(
            np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc),
            np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))

        coords = np.random.rand(N, 2)  # nodes' coordinates inside the community
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

            if comm_density:
                nb_edges = int(comm_density * M)
                tril_ind = np.tril_indices(com_siz, -1)
                indices = np.random.permutation(M)[:nb_edges]

                w_data[0] += [1] * nb_edges
                w_data[1][0] += [first_node + tril_ind[1][elem] for elem in indices]
                w_data[1][1] += [first_node + tril_ind[0][elem] for elem in indices]

            elif k_neigh:
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
                new_point = np.random.randint(0, N, 2)
                if info['node_com'][min(new_point)] != info['node_com'][max(new_point)]:
                    inter_edges.add((min(new_point), max(new_point)))
        else:
            # use random permutation
            indices = np.random.permutation(M)[:nb_edges]
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
        tmp_w_data = deepcopy(w_data[1][0])
        w_data[1][0] += w_data[1][1]
        w_data[1][1] += tmp_w_data
        w_data[1] = tuple(w_data[1])

        W = sparse.coo_matrix(tuple(w_data), shape=(N, N))

        for key, value in {'Nc': Nc, 'info': info}.items():
            setattr(self, key, value)

        super(Community, self).__init__(W=W, gtype='Community', coords=coords, **kwargs)
