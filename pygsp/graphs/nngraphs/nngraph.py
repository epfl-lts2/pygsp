import traceback

import numpy as np
from scipy import sparse, spatial
from scipy.spatial import distance

from pygsp import utils

from ..graph import Graph  # prevent circular import in Python < 3.5

_logger = utils.build_logger(__name__)


def _import_pfl():
    try:
        import pyflann as pfl
    except Exception as e:
        raise ImportError(
            "Cannot import pyflann. Choose another nearest "
            "neighbors method or try to install it with "
            "pip (or conda) install pyflann (or pyflann3). "
            "Original exception: {}".format(e)
        )
    return pfl


def _import_sklearn_neighbors():
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError(
            "Cannot import sklearn.neighbors. Install scikit-learn "
            "for fallback nearest neighbors support: "
            "pip install scikit-learn. "
            "Original exception: {}".format(e)
        )
    return NearestNeighbors


class NNGraph(Graph):
    r"""Nearest-neighbor graph from given point cloud.

    Parameters
    ----------
    Xin : ndarray
        Input points, Should be an `N`-by-`d` matrix, where `N` is the number
        of nodes in the graph and `d` is the dimension of the feature space.
    NNtype : string, optional
        Type of nearest neighbor graph to create. The options are 'knn' for
        k-Nearest Neighbors or 'radius' for epsilon-Nearest Neighbors (default
        is 'knn').
    use_flann : bool, optional
        Use Fast Library for Approximate Nearest Neighbors (FLANN) or not.
        (default is False)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width of the similarity kernel.
        By default, it is set to the average of the nearest neighbor distance.
    epsilon : float, optional
        Radius for the epsilon-neighborhood search (default is 0.01)
    plotting : dict, optional
        Dictionary of plotting parameters. See :obj:`pygsp.plotting`.
        (default is {})
    symmetrize_type : string, optional
        Type of symmetrization to use for the adjacency matrix. See
        :func:`pygsp.utils.symmetrization` for the options.
        (default is 'average')
    dist_type : string, optional
        Type of distance to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
        (default is 'euclidean')
    order : float, optional
        Only used if dist_type is 'minkowski'; represents the order of the
        Minkowski distance. (default is 0)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.default_rng(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(X)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    """

    def __init__(
        self,
        Xin,
        NNtype="knn",
        use_flann=False,
        center=True,
        rescale=True,
        k=10,
        sigma=None,
        epsilon=0.01,
        plotting={},
        symmetrize_type="average",
        dist_type="euclidean",
        order=0,
        **kwargs,
    ):
        self.Xin = Xin
        self.NNtype = NNtype
        self.use_flann = use_flann
        self.center = center
        self.rescale = rescale
        self.k = k
        self.sigma = sigma
        self.epsilon = epsilon
        self.symmetrize_type = symmetrize_type
        self.dist_type = dist_type
        self.order = order

        N, d = np.shape(self.Xin)
        Xout = self.Xin

        if k >= N:
            raise ValueError(
                "The number of neighbors (k={}) must be smaller "
                "than the number of nodes ({}).".format(k, N)
            )

        if self.center:
            Xout = self.Xin - np.kron(np.ones((N, 1)), np.mean(self.Xin, axis=0))

        if self.rescale:
            bounding_radius = 0.5 * np.linalg.norm(
                np.amax(Xout, axis=0) - np.amin(Xout, axis=0), 2
            )
            scale = np.power(N, 1.0 / float(min(d, 3))) / 10.0
            Xout *= scale / bounding_radius

        # Translate distance type string to corresponding Minkowski order.
        dist_translation = {
            "euclidean": 2,
            "manhattan": 1,
            "max_dist": np.inf,
            "minkowski": order,
        }

        if self.NNtype == "knn":
            spi = np.zeros(N * k)
            spj = np.zeros(N * k)
            spv = np.zeros(N * k)

            if self.use_flann:
                # Try pyflann first
                try:
                    pfl = _import_pfl()
                    pfl.set_distance_type(dist_type, order=order)
                    flann = pfl.FLANN()

                    # Default FLANN parameters (I tried changing the algorithm and
                    # testing performance on huge matrices, but the default one
                    # seems to work best).
                    NN, D = flann.nn(
                        Xout, Xout, num_neighbors=(k + 1), algorithm="kdtree"
                    )
                    _logger.debug("Using pyflann for k-NN search")

                except ImportError:
                    _logger.debug("pyflann not available, trying scikit-learn fallback")
                    try:
                        # Fallback to scikit-learn
                        NearestNeighbors = _import_sklearn_neighbors()

                        # Map distance types to sklearn metrics
                        sklearn_metrics = {
                            "euclidean": "euclidean",
                            "manhattan": "manhattan",
                            "max_dist": "chebyshev",
                            "minkowski": "minkowski",
                        }

                        metric = sklearn_metrics.get(dist_type, "euclidean")
                        metric_params = {}
                        if dist_type == "minkowski":
                            p_value = dist_translation[dist_type]
                            # Ensure p is valid for sklearn (must be > 0)
                            if p_value <= 0:
                                metric = (
                                    "euclidean"  # Fallback to euclidean for invalid p
                                )
                            else:
                                metric_params["p"] = p_value

                        nbrs = NearestNeighbors(
                            n_neighbors=k + 1,
                            algorithm="auto",
                            metric=metric,
                            metric_params=metric_params,
                        ).fit(Xout)
                        D, NN = nbrs.kneighbors(Xout)
                        _logger.debug("Using scikit-learn for k-NN search")

                    except ImportError:
                        _logger.debug(
                            "scikit-learn not available, falling back to scipy"
                        )
                        # Final fallback to scipy
                        kdt = spatial.KDTree(Xout)
                        D, NN = kdt.query(
                            Xout, k=(k + 1), p=dist_translation[dist_type]
                        )
                        _logger.debug("Using scipy KDTree for k-NN search")

            else:
                kdt = spatial.KDTree(Xout)
                D, NN = kdt.query(Xout, k=(k + 1), p=dist_translation[dist_type])
                _logger.debug("Using scipy KDTree for k-NN search")

            if self.sigma is None:
                self.sigma = np.mean(D[:, 1:])  # Discard distance to self.

            for i in range(N):
                spi[i * k : (i + 1) * k] = np.kron(np.ones(k), i)
                spj[i * k : (i + 1) * k] = NN[i, 1:]
                spv[i * k : (i + 1) * k] = np.exp(
                    -np.power(D[i, 1:], 2) / float(self.sigma)
                )

        elif self.NNtype == "radius":
            kdt = spatial.KDTree(Xout)
            # Use query_ball_point for radius-based neighbor search
            NN = [
                kdt.query_ball_point(point, r=epsilon, p=dist_translation[dist_type])
                for point in Xout
            ]
            # Compute distances for the neighbors found
            D = []
            for i, neighbors in enumerate(NN):
                if len(neighbors) > 0:
                    distances = [
                        distance.minkowski(
                            Xout[i], Xout[j], p=dist_translation[dist_type]
                        )
                        for j in neighbors
                    ]
                    D.append(distances)
                else:
                    D.append([])

            if self.sigma is None:
                # Calculate mean distance excluding self-references
                all_distances = []
                for i, neighbors in enumerate(NN):
                    for idx, j in enumerate(neighbors):
                        if j != i:  # Exclude self-reference
                            all_distances.append(D[i][idx])

                if all_distances:
                    self.sigma = np.mean(all_distances)
                else:
                    # If no neighbors found, set a default sigma
                    raise ValueError("No neighbors found")
            count = 0
            for i in range(N):
                # Count neighbors excluding self
                count = count + len([j for j in NN[i] if j != i])

            spi = np.zeros(count)
            spj = np.zeros(count)
            spv = np.zeros(count)

            start = 0
            for i in range(N):
                # Remove self-reference (index i) from neighbors
                neighbors_no_self = [j for j in NN[i] if j != i]
                distances_no_self = [D[i][idx] for idx, j in enumerate(NN[i]) if j != i]

                leng = len(neighbors_no_self)
                if leng > 0:
                    spi[start : start + leng] = np.kron(np.ones(leng), i)
                    spj[start : start + leng] = neighbors_no_self
                    spv[start : start + leng] = np.exp(
                        -np.power(distances_no_self, 2) / float(self.sigma)
                    )
                    start = start + leng

        else:
            raise ValueError(f"Unknown NNtype {self.NNtype}")

        W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

        # Sanity check
        if np.shape(W)[0] != np.shape(W)[1]:
            raise ValueError("Weight matrix W is not square")

        # Enforce symmetry. Note that checking symmetry with
        # np.abs(W - W.T).sum() is as costly as the symmetrization itself.
        W = utils.symmetrize(W, method=symmetrize_type)

        super().__init__(W, plotting=plotting, coords=Xout, **kwargs)

    def _get_extra_repr(self):
        return {
            "NNtype": self.NNtype,
            "use_flann": self.use_flann,
            "center": self.center,
            "rescale": self.rescale,
            "k": self.k,
            "sigma": f"{self.sigma:.2f}",
            "epsilon": f"{self.epsilon:.2f}",
            "symmetrize_type": self.symmetrize_type,
            "dist_type": self.dist_type,
            "order": self.order,
        }
