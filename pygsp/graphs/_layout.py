# -*- coding: utf-8 -*-

import numpy as np


class LayoutMixIn(object):

    def set_coordinates(self, kind='spring', seed=None, **kwargs):
        r"""Set node's coordinates (their position when plotting).

        Parameters
        ----------
        kind : string or array_like
            Kind of coordinates to generate. It controls the position of the
            nodes when plotting the graph. Can either pass an array of size Nx2
            or Nx3 to set the coordinates manually or the name of a layout
            algorithm. Available algorithms: community2D, random2D, random3D,
            ring2D, line1D, spring, laplacian_eigenmap2D, laplacian_eigenmap3D.
            Default is 'spring'.
        seed : int
            Seed for the random number generator when `kind` is `'random'`,
            `'community'`, or `'spring'`.
        kwargs : dict
            Additional parameters to be passed to the Fruchterman-Reingold
            force-directed algorithm when kind is spring.

        Examples
        --------
        >>> G = graphs.ErdosRenyi()
        >>> G.set_coordinates()
        >>> fig, ax = G.plot()

        """

        if not isinstance(kind, str):
            coords = np.asanyarray(kind).squeeze()
            check_1d = (coords.ndim == 1)
            check_2d_3d = (coords.ndim == 2) and (2 <= coords.shape[1] <= 3)
            if coords.shape[0] != self.N or not (check_1d or check_2d_3d):
                raise ValueError('Expecting coordinates to be of size N, Nx2, '
                                 'or Nx3.')
            self.coords = coords

        elif kind == 'line1D':
            self.coords = np.arange(self.N)

        elif kind == 'line2D':
            x, y = np.arange(self.N), np.zeros(self.N)
            self.coords = np.stack([x, y], axis=1)

        elif kind == 'ring2D':
            angle = np.arange(self.N) * 2 * np.pi / self.N
            self.coords = np.stack([np.cos(angle), np.sin(angle)], axis=1)

        elif kind == 'random2D':
            rng = np.random.default_rng(seed)
            self.coords = rng.uniform(size=(self.N, 2))

        elif kind == 'random3D':
            rng = np.random.default_rng(seed)
            self.coords = rng.uniform(size=(self.N, 3))

        elif kind == 'spring':
            self.coords = self._fruchterman_reingold(seed=seed, **kwargs)

        elif kind == 'community2D':
            if not hasattr(self, 'info') or 'node_com' not in self.info:
                ValueError('Missing arguments to the graph to be able to '
                           'compute community coordinates.')

            if 'world_rad' not in self.info:
                self.info['world_rad'] = np.sqrt(self.N)

            if 'comm_sizes' not in self.info:
                counts = Counter(self.info['node_com'])
                self.info['comm_sizes'] = np.array([cnt[1] for cnt
                                                    in sorted(counts.items())])

            Nc = self.info['comm_sizes'].shape[0]

            self.info['com_coords'] = self.info['world_rad'] * \
                np.array(list(zip(
                    np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc),
                    np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))

            # Coordinates of the nodes inside their communities
            rng = np.random.default_rng(seed)
            coords = rng.uniform(size=(self.N, 2))
            self.coords = np.array([[elem[0] * np.cos(2 * np.pi * elem[1]),
                                     elem[0] * np.sin(2 * np.pi * elem[1])]
                                    for elem in coords])

            for i in range(self.N):
                # Set coordinates as an offset from the center of the community
                # it belongs to
                comm_idx = self.info['node_com'][i]
                comm_rad = np.sqrt(self.info['comm_sizes'][comm_idx])
                self.coords[i] = self.info['com_coords'][comm_idx] + \
                    comm_rad * self.coords[i]
        elif kind == 'laplacian_eigenmap2D':
            self.compute_fourier_basis(n_eigenvectors=3)
            self.coords = self.U[:, 1:3]
        elif kind == 'laplacian_eigenmap3D':
            self.compute_fourier_basis(n_eigenvectors=4)
            self.coords = self.U[:, 1:4]
        else:
            raise ValueError('Unexpected argument kind={}.'.format(kind))

    def _fruchterman_reingold(self, dim=2, k=None, pos=None, fixed=[],
                              iterations=50, scale=1.0, center=None,
                              seed=None):
        # TODO doc
        # fixed: list of nodes with fixed coordinates
        # Position nodes using Fruchterman-Reingold force-directed algorithm.

        if center is None:
            center = np.zeros((1, dim))

        if np.shape(center)[1] != dim:
            self.logger.error('Spring coordinates: center has wrong size.')
            center = np.zeros((1, dim))

        if pos is None:
            dom_size = 1
            pos_arr = None
        else:
            # Determine size of existing domain to adjust initial positions
            dom_size = np.max(pos)
            rng = np.random.default_rng(seed)
            pos_arr = rng.uniform(size=(self.N, dim))
            pos_arr = pos_arr * dom_size + center
            for i in range(self.N):
                pos_arr[i] = np.asanyarray(pos[i])

        if k is None and len(fixed) > 0:
            # We must adjust k by domain size for layouts that are not near 1x1
            k = dom_size / np.sqrt(self.N)

        pos = _sparse_fruchterman_reingold(self.A, dim, k, pos_arr,
                                           fixed, iterations, seed)

        if len(fixed) == 0:
            pos = _rescale_layout(pos, scale=scale) + center

        return pos


def _sparse_fruchterman_reingold(A, dim, k, pos, fixed, iterations, seed):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    nnodes = A.shape[0]

    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except Exception:
        A = (sparse.coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        rng = np.random.default_rng(seed)
        pos = rng.uniform(size=(nnodes, dim))

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)

    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    t = 0.1
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(nnodes):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = A[i, :].toarray()
            # displacement "force"
            displacement[:, i] += \
                (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        pos += (displacement * t / length).T
        # cool temperature
        t -= dt

    return pos


def _rescale_layout(pos, scale=1):
    # rescale to (-scale, scale) in all axes

    # shift origin to (0,0)
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(pos[:, i].max(), lim)
    # rescale to (-scale,scale) in all directions, preserves aspect
    for i in range(pos.shape[1]):
        pos[:, i] *= scale / lim
    return pos
