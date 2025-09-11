import numpy as np
from scipy import sparse

from pygsp import utils

from .graph import Graph  # prevent circular import in Python < 3.5


class StochasticBlockModel(Graph):
    r"""Stochastic Block Model (SBM).

    The Stochastic Block Model graph is constructed by connecting nodes with a
    probability which depends on the cluster of the two nodes.  One can define
    the clustering association of each node, denoted by vector z, but also the
    probability matrix M.  All edge weights are equal to 1. By default, Mii >
    Mjk and nodes are uniformly clusterized.

    Parameters
    ----------
    N : int
        Number of nodes (default is 1024).
    k : float
        Number of classes (default is 5).
    z : array_like
        the vector of length N containing the association between nodes and
        classes (default is random uniform).
    M : array_like
        the k by k matrix containing the probability of connecting nodes based
        on their class belonging (default using p and q).
    p : float or array_like
        the diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a length k vector (default is p = 0.7).
    q : float or array_like
        the off-diagonal value(s) for the matrix M. If scalar they all have the
        same value. Otherwise expect a k x k matrix, diagonal will be
        discarded (default is q = 0.3/k).
    directed : bool
        Allow directed edges if True (default is False).
    self_loops : bool
        Allow self loops if True (default is False).
    connected : bool
        Force the graph to be connected (default is False).
    n_try : int or None
        Maximum number of trials to get a connected graph. If None, it will try
        forever.
    seed : int
        Seed for the random number generator (for reproducible graphs).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> G = graphs.StochasticBlockModel(
    ...     100, k=3, p=[0.4, 0.6, 0.3], q=0.02, seed=42)
    >>> G.set_coordinates(kind='spring', seed=42)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=0.8)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(
        self,
        N=1024,
        k=5,
        z=None,
        M=None,
        p=0.7,
        q=None,
        directed=False,
        self_loops=False,
        connected=False,
        n_try=10,
        seed=None,
        **kwargs,
    ):
        self.k = k
        self.directed = directed
        self.self_loops = self_loops
        self.connected = connected
        self.n_try = n_try
        self.seed = seed

        rng = np.random.default_rng(seed)

        if z is None:
            z = rng.integers(0, k, N)
            z.sort()  # Sort for nice spy plot of W, where blocks are apparent.
        self.z = z

        if M is None:
            self.p = p
            p = np.asanyarray(p)
            if p.size == 1:
                p = p * np.ones(k)
            if p.shape != (k,):
                raise ValueError(
                    "Optional parameter p is neither a scalar "
                    "nor a vector of length k."
                )

            if q is None:
                q = 0.3 / k
            self.q = q
            q = np.asanyarray(q)
            if q.size == 1:
                q = q * np.ones((k, k))
            if q.shape != (k, k):
                raise ValueError(
                    "Optional parameter q is neither a scalar "
                    "nor a matrix of size k x k."
                )

            M = q
            M.flat[:: k + 1] = p  # edit the diagonal terms

        self.M = M

        if (M < 0).any() or (M > 1).any():
            raise ValueError("Probabilities should be in [0, 1].")

        # TODO: higher memory, lesser computation alternative.
        # Along the lines of np.random.uniform(size=(N, N)) < p.
        # Or similar to sparse.random(N, N, p, data_rvs=lambda n: np.ones(n)).

        while (n_try is None) or (n_try > 0):
            nb_row, nb_col = 0, 0
            csr_data, csr_i, csr_j = [], [], []
            for _ in range(N**2):
                if nb_row != nb_col or self_loops:
                    if nb_row >= nb_col or directed:
                        if rng.uniform() < M[z[nb_row], z[nb_col]]:
                            csr_data.append(1)
                            csr_i.append(nb_row)
                            csr_j.append(nb_col)
                if nb_row < N - 1:
                    nb_row += 1
                else:
                    nb_row = 0
                    nb_col += 1

            W = sparse.csr_matrix((csr_data, (csr_i, csr_j)), shape=(N, N))

            if not directed:
                W = utils.symmetrize(W, method="tril")

            if not connected:
                break
            if Graph(W).is_connected():
                break
            if n_try is not None:
                n_try -= 1
        if connected and n_try == 0:
            raise ValueError(
                "The graph could not be connected after {} "
                "trials. Increase the connection probability "
                "or the number of trials.".format(self.n_try)
            )

        self.info = {
            "node_com": z,
            "comm_sizes": np.bincount(z),
            "world_rad": np.sqrt(N),
        }

        super().__init__(W, **kwargs)

    def _get_extra_repr(self):
        attrs = {"k": self.k}
        if type(self.p) is float:
            attrs["p"] = f"{self.p:.2f}"
        if type(self.q) is float:
            attrs["q"] = f"{self.q:.2f}"
        attrs.update(
            {
                "directed": self.directed,
                "self_loops": self.self_loops,
                "connected": self.connected,
                "seed": self.seed,
            }
        )
        return attrs
