from functools import partial

import numpy as np

from pygsp import utils

from ..graphs import Graph

# prevent circular import in Python < 3.5
from . import approximations

_logger = utils.build_logger(__name__)


class Filter:
    r"""
    The base Filter class.

    * Provide a common interface (and implementation) to filter objects.
    * Can be instantiated to construct custom filters from functions.
    * Initialize attributes for derived classes.

    Parameters
    ----------
    G : graph
        The graph to which the filter bank is tailored.
    kernels : function or list of functions
        A (list of) function(s) defining the filter bank in the Fourier domain.
        One function per filter.

    Attributes
    ----------
    G : Graph
        The graph to which the filter bank was tailored. It is a reference to
        the graph passed when instantiating the class.
    n_features_in : int
        Number of signals or features the filter bank takes in.
    n_features_out : int
        Number of signals or features the filter bank gives out.
    n_filters : int
        Number of filters in the filter bank.

    Examples
    --------
    >>> G = graphs.Logo()
    >>> my_filter = filters.Filter(G, lambda x: x / (1. + x))
    >>>
    >>> # Signal: Kronecker delta.
    >>> signal = np.zeros(G.N)
    >>> signal[42] = 1
    >>>
    >>> filtered_signal = my_filter.filter(signal)

    """

    def __init__(self, G, kernels):
        self.G = G

        try:
            iter(kernels)
        except TypeError:
            kernels = [kernels]
        self._kernels = kernels

        # Only used by subclasses to instantiate a single filterbank.
        self.n_features_in, self.n_features_out = (1, len(kernels))
        self.shape = (self.n_features_out, self.n_features_in)
        self.n_filters = self.n_features_in * self.n_features_out
        self.Nf = self.n_filters  # TODO: kept for backward compatibility only.

    def _get_extra_repr(self):
        """To be overloaded by children."""
        return dict()

    def __repr__(self):
        attrs = {"in": self.n_features_in, "out": self.n_features_out}
        attrs.update(self._get_extra_repr())
        s = ""
        for key, value in attrs.items():
            s += f"{key}={value}, "
        return f"{self.__class__.__name__}({s[:-2]})"

    def __len__(self):
        # Numpy returns shape[0].
        return self.n_filters

    def __getitem__(self, key):
        return Filter(self.G, self._kernels[key])

    def __add__(self, other):
        """Concatenation of filterbanks."""
        if not isinstance(other, Filter):
            return NotImplemented
        return Filter(self.G, self._kernels + other._kernels)

    def __call__(self, x):
        if isinstance(x, Graph):
            return Filter(x, self._kernels)
        else:
            return self.evaluate(x)

    def __matmul__(self, other):
        return self.filter(other)

    def toarray(self):
        r"""Return an array representation of the filter bank.

        See :meth:`compute_frame`.
        """
        return self.compute_frame()

    def evaluate(self, x):
        r"""Evaluate the kernels at given frequencies.

        Parameters
        ----------
        x : array_like
            Graph frequencies at which to evaluate the filter.

        Returns
        -------
        y : ndarray
            Frequency response of the filters. Shape ``(g.Nf, len(x))``.

        Examples
        --------
        Frequency response of a low-pass filter:

        >>> import matplotlib.pyplot as plt
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> f = filters.Expwin(G)
        >>> G.compute_fourier_basis()
        >>> y = f.evaluate(G.e)
        >>> plt.plot(G.e, y[0])  # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object at ...>]

        """
        x = np.asanyarray(x)
        # Avoid to copy data as with np.array([g(x) for g in self._kernels]).
        y = np.empty([self.Nf] + list(x.shape))
        for i, kernel in enumerate(self._kernels):
            y[i] = kernel(x)
        return y

    def filter(self, s, method="chebyshev", order=30):
        r"""Filter signals (analysis or synthesis).

        A signal is defined as a rank-3 tensor of shape ``(N_NODES, N_SIGNALS,
        N_FEATURES)``, where ``N_NODES`` is the number of nodes in the graph,
        ``N_SIGNALS`` is the number of independent signals, and ``N_FEATURES``
        is the number of features which compose a graph signal, or the
        dimensionality of a graph signal. For example if you filter a signal
        with a filter bank of 8 filters, you're extracting 8 features and
        decomposing your signal into 8 parts. That is called analysis. Your are
        thus transforming your signal tensor from ``(G.N, 1, 1)`` to ``(G.N, 1,
        8)``. Now you may want to combine back the features to form an unique
        signal. For this you apply again 8 filters, one filter per feature, and
        sum the result up. As such you're transforming your ``(G.N, 1, 8)``
        tensor signal back to ``(G.N, 1, 1)``. That is known as synthesis. More
        generally, you may want to map a set of features to another, though
        that is not implemented yet.

        The method computes the transform coefficients of a signal :math:`s`,
        where the atoms of the transform dictionary are generalized
        translations of each graph spectral filter to each vertex on the graph:

        .. math:: c = D^* s,

        where the columns of :math:`D` are :math:`g_{i,m} = T_i g_m` and
        :math:`T_i` is a generalized translation operator applied to each
        filter :math:`\hat{g}_m(\cdot)`. Each column of :math:`c` is the
        response of the signal to one filter.

        In other words, this function is applying the analysis operator
        :math:`D^*`, respectively the synthesis operator :math:`D`, associated
        with the frame defined by the filter bank to the signals.

        Parameters
        ----------
        s : array_like
            Graph signals, a tensor of shape ``(N_NODES, N_SIGNALS,
            N_FEATURES)``, where ``N_NODES`` is the number of nodes in the
            graph, ``N_SIGNALS`` the number of independent signals you want to
            filter, and ``N_FEATURES`` is either 1 (analysis) or the number of
            filters in the filter bank (synthesis).
        method : {'exact', 'chebyshev'}
            Whether to use the exact method (via the graph Fourier transform)
            or the Chebyshev polynomial approximation. A Lanczos
            approximation is coming.
        order : int
            Degree of the Chebyshev polynomials.

        Returns
        -------
        s : ndarray
            Graph signals, a tensor of shape ``(N_NODES, N_SIGNALS,
            N_FEATURES)``, where ``N_NODES`` and ``N_SIGNALS`` are the number
            of nodes and signals of the signal tensor that pas passed in, and
            ``N_FEATURES`` is either 1 (synthesis) or the number of filters in
            the filter bank (analysis).

        References
        ----------
        See :cite:`hammond2011wavelets` for details on filtering graph signals.

        Examples
        --------

        Create a bunch of smooth signals by low-pass filtering white noise:

        >>> import matplotlib.pyplot as plt
        >>> G = graphs.Ring(N=60)
        >>> G.estimate_lmax()
        >>> s = np.random.default_rng(42).uniform(size=(G.N, 10))
        >>> taus = [1, 10, 100]
        >>> s = filters.Heat(G, taus).filter(s)
        >>> s.shape
        (60, 10, 3)

        Plot the 3 smoothed versions of the 10th signal:

        >>> fig, ax = plt.subplots()
        >>> G.set_coordinates('line1D')  # To visualize multiple signals in 1D.
        >>> _ = G.plot(s[:, 9, :], ax=ax)
        >>> legend = [r'$\tau={}$'.format(t) for t in taus]
        >>> ax.legend(legend)  # doctest: +ELLIPSIS
        <matplotlib.legend.Legend object at ...>

        Low-pass filter a delta to create a localized smooth signal:

        >>> G = graphs.Sensor(30, seed=42)
        >>> G.compute_fourier_basis()  # Reproducible computation of lmax.
        >>> s1 = np.zeros(G.N)
        >>> s1[13] = 1
        >>> s1 = filters.Heat(G, 3).filter(s1)
        >>> s1.shape
        (30,)

        Filter and reconstruct our signal:

        >>> g = filters.MexicanHat(G, Nf=4)
        >>> s2 = g.analyze(s1)
        >>> s2.shape
        (30, 4)
        >>> s2 = g.synthesize(s2)
        >>> s2.shape
        (30,)

        Look how well we were able to reconstruct:

        >>> fig, axes = plt.subplots(1, 2)
        >>> _ = G.plot(s1, ax=axes[0])
        >>> _ = G.plot(s2, ax=axes[1])
        >>> print('{:.5f}'.format(np.linalg.norm(s1 - s2)))
        0.27649

        Perfect reconstruction with Itersine, a tight frame:

        >>> g = filters.Itersine(G)
        >>> s2 = g.analyze(s1, method='exact')
        >>> s2 = g.synthesize(s2, method='exact')
        >>> np.linalg.norm(s1 - s2) < 1e-10
        True

        """
        s = self.G._check_signal(s)

        # TODO: not in self.Nin (Nf = Nin x Nout).
        if s.ndim == 1 or s.shape[-1] not in [1, self.Nf]:
            if s.ndim == 3:
                raise ValueError(
                    "Third dimension (#features) should be "
                    "either 1 or the number of filters Nf = {}, "
                    "got {}.".format(self.Nf, s.shape)
                )
            s = np.expand_dims(s, -1)
        n_features_in = s.shape[-1]

        if s.ndim < 3:
            s = np.expand_dims(s, 1)
        n_signals = s.shape[1]

        if s.ndim > 3:
            raise ValueError("At most 3 dimensions: " "#nodes x #signals x #features.")
        assert s.ndim == 3

        # TODO: generalize to 2D (m --> n) filter banks.
        # Only 1 --> Nf (analysis) and Nf --> 1 (synthesis) for now.
        n_features_out = self.Nf if n_features_in == 1 else 1

        if method == "exact":
            # TODO: will be handled by g.adjoint().
            axis = 1 if n_features_in == 1 else 2
            f = self.evaluate(self.G.e)
            f = np.expand_dims(f.T, axis)
            assert f.shape == (self.G.N, n_features_in, n_features_out)

            s = self.G.gft(s)
            s = np.matmul(s, f)
            s = self.G.igft(s)

        elif method == "chebyshev":
            # TODO: update Chebyshev implementation (after 2D filter banks).
            c = approximations.compute_cheby_coeff(self, m=order)

            if n_features_in == 1:  # Analysis.
                s = s.squeeze(axis=2)
                s = approximations.cheby_op(self.G, c, s)
                s = s.reshape((self.G.N, n_features_out, n_signals), order="F")
                s = s.swapaxes(1, 2)

            elif n_features_in == self.Nf:  # Synthesis.
                s = s.swapaxes(1, 2)
                s_in = s.reshape((self.G.N * n_features_in, n_signals), order="F")
                s = np.zeros((self.G.N, n_signals))
                tmpN = np.arange(self.G.N, dtype=int)
                for i in range(n_features_in):
                    s += approximations.cheby_op(
                        self.G, c[i], s_in[i * self.G.N + tmpN]
                    )
                s = np.expand_dims(s, 2)

        else:
            raise ValueError(f"Unknown method {method}.")

        # Return a 1D signal if e.g. a 1D signal was filtered by one filter.
        return s.squeeze()

    def analyze(self, s, method="chebyshev", order=30):
        r"""Convenience alias to :meth:`filter`."""
        if s.ndim == 3 and s.shape[-1] != 1:
            raise ValueError(
                "Last dimension (#features) should be " "1, got {}.".format(s.shape)
            )
        return self.filter(s, method, order)

    def synthesize(self, s, method="chebyshev", order=30):
        r"""Convenience wrapper around :meth:`filter`.

        Will be an alias to `adjoint().filter()` in the future.
        """
        if s.shape[-1] != self.Nf:
            raise ValueError(
                "Last dimension (#features) should be the number "
                "of filters Nf = {}, got {}.".format(self.Nf, s.shape)
            )
        return self.filter(s, method, order)

    def localize(self, i, **kwargs):
        r"""Localize the kernels at a node (to visualize them).

        That is particularly useful to visualize a filter in the vertex domain.

        A kernel is localized on vertex :math:`v_i` by filtering a Kronecker
        delta :math:`\delta_i` as

        .. math:: (g(L) \delta_i)(j) = g(L)(i,j),
                  \text{ where } \delta_i(j) =
                  \begin{cases} 0 \text{ if } i \neq j, \\
                                1 \text{ if } i = j.    \end{cases}

        Parameters
        ----------
        i : int
            Index of the node where to localize the kernel.
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        s : ndarray
            Kernel localized at vertex i.

        Examples
        --------
        Visualize heat diffusion on a grid by localizing the heat kernel.

        >>> import matplotlib
        >>> N = 20
        >>> DELTA = N//2 * (N+1)
        >>> G = graphs.Grid2d(N)
        >>> G.estimate_lmax()
        >>> g = filters.Heat(G, 100)
        >>> s = g.localize(DELTA)
        >>> _ = G.plot(s, highlight=DELTA)

        """
        s = np.zeros(self.G.N)
        s[i] = 1
        return np.sqrt(self.G.N) * self.filter(s, **kwargs)

    def estimate_frame_bounds(self, x=None):
        r"""Estimate lower and upper frame bounds.

        A filter bank forms a frame if there are positive real numbers
        :math:`A` and :math:`B`, :math:`0 < A \leq B < \infty`, that satisfy
        the *frame condition*

        .. math:: A \|x\|^2 \leq \| g(L) x \|^2 \leq B \|x\|^2

        for all signals :math:`x \in \mathbb{R}^N`, where :math:`g(L)` is the
        analysis operator of the filter bank.

        As :math:`g(L) = U g(\Lambda) U^\top` is diagonalized by the Fourier
        basis :math:`U` with eigenvalues :math:`\Lambda`, :math:`\| g(L) x \|^2
        = \| g(\Lambda) U^\top x \|^2`, and :math:`A = \min g^2(\Lambda)`,
        :math:`B = \max g^2(\Lambda)`.

        Parameters
        ----------
        x : array_like
            Graph frequencies at which to evaluate the filter bank `g(x)`.
            The default is ``x = np.linspace(0, G.lmax, 1000)``.
            The exact bounds are given by evaluating the filter bank at the
            eigenvalues of the graph Laplacian, i.e., ``x = G.e``.

        Returns
        -------
        A : float
            Lower frame bound of the filter bank.
        B : float
            Upper frame bound of the filter bank.

        See Also
        --------
        compute_frame: compute the frame
        complement: complement a filter bank to become a tight frame

        Examples
        --------

        >>> from matplotlib import pyplot as plt
        >>> fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        >>> G = graphs.Sensor(64, seed=42)
        >>> G.compute_fourier_basis()
        >>> g = filters.Abspline(G, 7)

        Estimation quality vs speed (loose & fast -> exact & slow):

        >>> A, B = g.estimate_frame_bounds(np.linspace(0, G.lmax, 5))
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.883, B=2.288
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.708, B=2.359
        >>> A, B = g.estimate_frame_bounds(G.e)
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.875, B=2.359

        The frame bounds can be seen in the plot of the filter bank as the
        minimum and maximum of their squared sum (the black curve):

        >>> def plot(g, ax):
        ...     g.plot(ax=ax, labels=False, title='')
        ...     ax.hlines(B, 0, G.lmax, colors='r', zorder=3,
        ...               label='upper bound $B={:.2f}$'.format(B))
        ...     ax.hlines(A, 0, G.lmax, colors='b', zorder=3,
        ...               label='lower bound $A={:.2f}$'.format(A))
        ...     ax.legend(loc='center right')
        >>> plot(g, axes[0, 0])

        The heat kernel has a null-space and doesn't define a frame (the lower
        bound should be greater than 0 to have a frame):

        >>> g = filters.Heat(G)
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.000, B=1.000
        >>> plot(g, axes[0, 1])

        Without a null-space, the heat kernel forms a frame:

        >>> g = filters.Heat(G, scale=[1, 10])
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.135, B=2.000
        >>> plot(g, axes[1, 0])

        A kernel and its dual form a tight frame (A=B):

        >>> g = filters.Regular(G)
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.000, B=1.000
        >>> plot(g, axes[1, 1])
        >>> fig.tight_layout()

        The Itersine filter bank forms a tight frame (A=B):

        >>> g = filters.Itersine(G)
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.000, B=1.000

        """
        if x is None:
            x = np.linspace(0, self.G.lmax, 1000)
        else:
            x = np.asanyarray(x)

        sum_filters = np.sum(self.evaluate(x) ** 2, axis=0)

        return sum_filters.min(), sum_filters.max()

    def compute_frame(self, **kwargs):
        r"""Compute the associated frame.

        A filter bank defines a frame, which is a generalization of a basis to
        sets of vectors that may be linearly dependent. See
        `Wikipedia <https://en.wikipedia.org/wiki/Frame_(linear_algebra)>`_.

        The frame of a filter bank is the union of the frames of its
        constituent filters. The vectors forming the frame are the rows of the
        *analysis operator*

        .. math::
            g(L) = \begin{pmatrix} g_1(L) \\ \vdots \\ g_F(L) \end{pmatrix}
                   \in \mathbb{R}^{NF \times N}, \quad
            g_i(L) = U g_i(\Lambda) U^\top,

        where :math:`g_i` are the filter kernels, :math:`N` is the number of
        nodes, :math:`F` is the number of filters, :math:`L` is the graph
        Laplacian, :math:`\Lambda` is a diagonal matrix of the Laplacian's
        eigenvalues, and :math:`U` is the Fourier basis, i.e., its columns are
        the eigenvectors of the Laplacian.

        The matrix :math:`g(L)` represents the *analysis operator* of the
        frame. Its adjoint :math:`g(L)^\top` represents the *synthesis
        operator*. A signal :math:`x` is thus analyzed with the frame by
        :math:`y = g(L) x`, and synthesized from its frame coefficients by
        :math:`z = g(L)^\top y`. Computing this matrix is however a rather
        inefficient way of doing those operations.

        If :math:`F > 1`, the frame is said to be over-complete and the
        representation :math:`g(L) x` of the signal :math:`x` is said to be
        redundant.

        If the frame is tight, the *frame operator* :math:`g(L)^\top g(L)` is
        diagonal, with entries equal to the frame bound :math:`A = B`.

        Parameters
        ----------
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        frame : ndarray
            Array of size (#nodes x #filters) x #nodes.

        See Also
        --------
        estimate_frame_bounds: estimate the frame bounds
        filter: more efficient way to filter signals

        Examples
        --------

        >>> G = graphs.Sensor(100, seed=42)
        >>> G.compute_fourier_basis()

        Filtering as a multiplication with the matrix representation of the
        frame analysis operator:

        >>> g = filters.MexicanHat(G, Nf=6)
        >>> s = np.random.default_rng().uniform(size=G.N)
        >>>
        >>> gL = g.compute_frame()
        >>> gL.shape
        (600, 100)
        >>> s1 = gL.dot(s)
        >>> s1 = s1.reshape(G.N, -1, order='F')
        >>>
        >>> s2 = g.filter(s)
        >>> np.all(np.abs(s1 - s2) < 1e-10)
        True

        The frame operator of a tight frame is the identity matrix times the
        frame bound:

        >>> g = filters.Itersine(G)
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.000, B=1.000
        >>> gL = g.compute_frame(method='exact')
        >>> gL.shape
        (600, 100)
        >>> np.all(gL.T.dot(gL) - np.identity(G.N) < 1e-10)
        True

        """
        if self.G.N > 2000:
            _logger.warning(
                "Creating a big matrix. " "You should prefer the filter method."
            )

        # Filter one delta per vertex.
        s = np.identity(self.G.N)
        return self.filter(s, **kwargs).T.reshape(-1, self.G.N)

    def complement(self, frame_bound=None):
        r"""Return the filter that makes the frame tight.

        The complementary filter is designed such that the union of a filter
        bank and its complementary filter forms a tight frame.

        Parameters
        ----------
        frame_bound : float or None
            The desired frame bound :math:`A = B` of the resulting tight frame.
            The chosen bound should be larger than the sum of squared
            evaluations of all filters in the filter bank. If None (the
            default), the method chooses the smallest feasible bound.

        Returns
        -------
        complement: Filter
            The complementary filter.

        See Also
        --------
        estimate_frame_bounds: estimate the frame bounds

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> G = graphs.Sensor(100, seed=42)
        >>> G.estimate_lmax()
        >>> g = filters.Abspline(G, 4)
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=0.200, B=1.971
        >>> fig, axes = plt.subplots(1, 2)
        >>> fig, ax = g.plot(ax=axes[0])
        >>> g += g.complement()
        >>> A, B = g.estimate_frame_bounds()
        >>> print('A={:.3f}, B={:.3f}'.format(A, B))
        A=1.971, B=1.971
        >>> fig, ax = g.plot(ax=axes[1])

        """

        def kernel(x, *args, **kwargs):
            y = self.evaluate(x)
            np.power(y, 2, out=y)
            y = np.sum(y, axis=0)

            if frame_bound is None:
                bound = y.max()
            elif y.max() > frame_bound:
                raise ValueError(
                    "The chosen bound is not feasible. "
                    "Choose at least {}.".format(y.max())
                )
            else:
                bound = frame_bound

            return np.sqrt(bound - y)

        return Filter(self.G, kernel)

    def inverse(self):
        r"""Return the pseudo-inverse filter bank.

        The pseudo-inverse of the *analysis filter bank* :math:`g` is the
        *synthesis filter bank* :math:`g^+` such that

        .. math:: g(L)^+ g(L) = I,

        where :math:`I` is the identity matrix, and the *synthesis operator*

        .. math:: g(L)^+ = (g(L)\top g(L))^{-1} g(L)^\top
                         = (g_1(L)^+, \dots, g_F(L)^+)
                           \in \mathbb{R}^{N \times NF}

        is the left pseudo-inverse of the analysis operator :math:`g(L)`. Note
        that :math:`g_i(L)^+` is the pseudo-inverse of :math:`g_i(L)`,
        :math:`N` is the number of vertices, and :math:`F` is the number of
        filters in the bank.

        The above relation holds, and the reconstruction is exact, if and only
        if :math:`g(L)` is a frame. To be a frame, the rows of :math:`g(L)`
        must span the whole space (i.e., :math:`g(L)` must have full row rank).
        That is the case if the lower frame bound :math:`A > 0`. If
        :math:`g(L)` is not a frame, the reconstruction :math:`g(L)^+ g(L) x`
        will be the closest to :math:`x` in the least square sense.

        While there exists infinitely many inverses of the analysis operator of
        a frame, the pseudo-inverse is unique and corresponds to the *canonical
        dual* of the filter kernel.

        The *frame operator* of :math:`g^+` is :math:`g(L)^+ (g(L)^+)^\top =
        (g(L)\top g(L))^{-1}`, the inverse of the frame operator of :math:`g`.
        Similarly, its *frame bounds* are :math:`A^{-1}` and :math:`B^{-1}`,
        where :math:`A` and :math:`B` are the frame bounds of :math:`g`.

        If :math:`g` is tight (i.e., :math:`A=B`), the canonical dual is given
        by :math:`g^+ = g / A` (i.e., :math:`g^+_i = g_i / A \ \forall i`).

        Returns
        -------
        inverse : :class:`pygsp.filters.Filter`
            The pseudo-inverse filter bank, which synthesizes (or reconstructs)
            a signal from its coefficients using the canonical dual frame.

        See Also
        --------
        estimate_frame_bounds: estimate the frame bounds

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> G = graphs.Sensor(100, seed=42)
        >>> G.compute_fourier_basis()
        >>> # Create a filter and its inverse.
        >>> g = filters.Abspline(G, 5)
        >>> h = g.inverse()
        >>> # Plot them.
        >>> fig, axes = plt.subplots(1, 2)
        >>> _ = g.plot(ax=axes[0], title='original filter bank')
        >>> _ = h.plot(ax=axes[1], title='inverse filter bank')
        >>> # Filtering with the inverse reconstructs the original signal.
        >>> x = np.random.default_rng(42).normal(size=G.N)
        >>> y = g.filter(x, method='exact')
        >>> z = h.filter(y, method='exact')
        >>> np.linalg.norm(x - z) < 1e-10
        True
        >>> # Indeed, they cancel each others' effect.
        >>> Ag, Bg = g.estimate_frame_bounds()
        >>> Ah, Bh = h.estimate_frame_bounds()
        >>> print('A(g)*B(h) = {:.3f} * {:.3f} = {:.3f}'.format(Ag, Bh, Ag*Bh))
        A(g)*B(h) = 0.687 * 1.457 = 1.000
        >>> print('B(g)*A(h) = {:.3f} * {:.3f} = {:.3f}'.format(Bg, Ah, Bg*Ah))
        B(g)*A(h) = 1.994 * 0.501 = 1.000

        """

        A, B = self.estimate_frame_bounds()
        if A == 0:
            _logger.warning(
                "The filter bank is not invertible as it is not "
                "a frame (lower frame bound A=0)."
            )

        elif A / B < 1e-10:
            _logger.warning(
                "The filter bank is badly conditioned. "
                "The inverse will be approximate."
            )

        def kernel(g, i, x):
            y = g.evaluate(x).T
            z = np.linalg.pinv(np.expand_dims(y, axis=-1)).squeeze(axis=-2)
            return z[:, i]  # Return one filter.

        kernels = [partial(kernel, self, i) for i in range(self.n_filters)]

        return Filter(self.G, kernels)

    def plot(
        self,
        n=500,
        eigenvalues=None,
        sum=None,
        labels=None,
        title=None,
        ax=None,
        **kwargs,
    ):
        r"""Docstring overloaded at import time."""
        from pygsp.plotting import _plot_filter

        return _plot_filter(
            self,
            n=n,
            eigenvalues=eigenvalues,
            sum=sum,
            labels=labels,
            title=title,
            ax=ax,
            **kwargs,
        )
