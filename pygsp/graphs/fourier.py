# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse

from pygsp import utils


logger = utils.build_logger(__name__)


class GraphFourier(object):

    def _check_fourier_properties(self, name, desc):
        if not hasattr(self, '_' + name):
            self.logger.warning('The {} G.{} is not available, we need to '
                                'compute the Fourier basis. Explicitly call '
                                'G.compute_fourier_basis() once beforehand '
                                'to suppress the warning.'.format(desc, name))
            self.compute_fourier_basis()
        return getattr(self, '_' + name)

    @property
    def U(self):
        r"""Fourier basis (eigenvectors of the Laplacian).

        Is computed by :meth:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('U', 'Fourier basis')

    @property
    def e(self):
        r"""Eigenvalues of the Laplacian (square of graph frequencies).

        Is computed by :meth:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('e', 'eigenvalues vector')

    @property
    def coherence(self):
        r"""Coherence of the Fourier basis.

        The mutual coherence between the basis of Kronecker deltas on the graph
        and the basis of graph Laplacian eigenvectors is defined as

        .. math:: \mu = \max_{\ell,i} | \langle U_\ell, \delta_i \rangle |
                      = \max_{\ell,i} | U_{\ell, i} |
                      \in \left[ \frac{1}{\sqrt{N}}, 1 \right].

        It is a measure of the localization of the Fourier modes (Laplacian
        eigenvectors). The smaller the value, the more localized the
        eigenvectors can be. The extreme is a node that is disconnected from
        the rest of the graph: an eigenvector will be localized as a Kronecker
        delta there. In the classical setting, Fourier modes (which are complex
        exponentials) are completely delocalized, and the coherence equals one.

        The value is computed by :meth:`compute_fourier_basis`.

        Examples
        --------

        Delocalized eigenvectors.

        >>> graph = graphs.Path(100)
        >>> graph.compute_fourier_basis()
        >>> minimum = 1 / np.sqrt(graph.n_vertices)
        >>> print('{:.2f} in [{:.2f}, 1]'.format(graph.coherence, minimum))
        0.14 in [0.10, 1]
        >>>
        >>> # Plot some delocalized eigenvectors.
        >>> import matplotlib.pyplot as plt
        >>> graph.set_coordinates('line1D')
        >>> _ = graph.plot(graph.U[:, :5])

        Localized eigenvectors.

        >>> graph = graphs.Sensor(64, seed=20)
        >>> graph.compute_fourier_basis()
        >>> minimum = 1 / np.sqrt(graph.n_vertices)
        >>> print('{:.2f} in [{:.2f}, 1]'.format(graph.coherence, minimum))
        0.91 in [0.12, 1]
        >>>
        >>> # Plot the most localized eigenvector.
        >>> import matplotlib.pyplot as plt
        >>> idx = np.argmax(np.max(graph.U, axis=0))
        >>> _ = graph.plot(graph.U[:, idx])

        """
        return self._check_fourier_properties('coherence',
                                              'Fourier basis coherence')

    def compute_fourier_basis(self, n_eigenvectors=None, recompute=False):
        r"""Compute the (partial) Fourier basis of the graph (cached).

        The result is cached and accessible by the :attr:`U`, :attr:`e`,
        :attr:`lmax`, and :attr:`coherence` properties.

        Parameters
        ----------
        n_eigenvectors : int or `None`
            Number of eigenvectors to compute. If `None`, all eigenvectors
            are computed. (default: None)
        recompute: bool
            Force to recompute the Fourier basis if already existing.

        Notes
        -----
        'G.compute_fourier_basis()' computes a full eigendecomposition of
        the graph Laplacian :math:`L` such that:

        .. math:: L = U \Lambda U^*,

        or a partial eigendecomposition of the graph Laplacian :math:`L`
        such that:

        .. math:: L \approx U \Lambda U^*,

        where :math:`\Lambda` is a diagonal matrix of eigenvalues and the
        columns of :math:`U` are the eigenvectors.

        *G.e* is a vector of length `n_eigenvectors` :math:`\le` *G.N*
        containing the Laplacian eigenvalues. The largest eigenvalue is stored
        in *G.lmax*. The eigenvectors are stored as column vectors of *G.U* in
        the same order that the eigenvalues. Finally, the coherence of the
        Fourier basis is found in *G.coherence*.

        References
        ----------
        See :cite:`chung1997spectral`.

        Examples
        --------
        >>> G = graphs.Torus()
        >>> G.compute_fourier_basis(n_eigenvectors=64)
        >>> G.U.shape
        (256, 64)
        >>> G.e.shape
        (64,)
        >>> G.compute_fourier_basis()
        >>> G.U.shape
        (256, 256)
        >>> G.e.shape
        (256,)
        >>> G.lmax == G.e[-1]
        True
        >>> G.coherence < 1
        True

        """
        if n_eigenvectors is None:
            n_eigenvectors = self.N

        if (hasattr(self, '_e') and hasattr(self, '_U') and not recompute
                and n_eigenvectors <= len(self.e)):
            return

        assert self.L.shape == (self.N, self.N)
        if self.N**2 * n_eigenvectors > 3000**3:
            self.logger.warning(
                'Computing the {0} eigendecomposition of a large matrix ({1} x'
                ' {1}) is expensive. Consider decreasing n_eigenvectors '
                'or, if using the Fourier basis to filter, using a '
                'polynomial filter instead.'.format(
                    'full' if n_eigenvectors == self.N else 'partial',
                    self.N))

        # TODO: handle non-symmetric Laplacians. Test lap_type?
        if n_eigenvectors == self.N:
            self._e, self._U = np.linalg.eigh(self.L.toarray())
        else:
            # fast partial eigendecomposition of hermitian matrices
            self._e, self._U = sparse.linalg.eigsh(self.L,
                                                   n_eigenvectors,
                                                   which='SM')
        # Columns are eigenvectors. Sorted in ascending eigenvalue order.

        # Smallest eigenvalue should be zero: correct numerical errors.
        # Eigensolver might sometimes return small negative values, which
        # filter's implementations may not anticipate. Better for plotting too.
        assert -1e-5 < self._e[0] < 1e-5
        self._e[0] = 0

        # Bounded spectrum.
        assert self._e[-1] <= self._get_upper_bound() + 1e-5

        assert np.max(self._e) == self._e[-1]
        if n_eigenvectors == self.N:
            self._lmax = self._e[-1]
            self._coherence = np.max(np.abs(self._U))

    def gft(self, s):
        r"""Compute the graph Fourier transform.

        The graph Fourier transform of a signal :math:`s` is defined as

        .. math:: \hat{s} = U^* s,

        where :math:`U` is the Fourier basis attr:`U` and :math:`U^*` denotes
        the conjugate transpose or Hermitian transpose of :math:`U`.

        Parameters
        ----------
        s : ndarray
            Graph signal in the vertex domain.

        Returns
        -------
        s_hat : ndarray
            Representation of s in the Fourier domain.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> s = np.random.normal(size=(G.N, 5, 1))
        >>> s_hat = G.gft(s)
        >>> s_star = G.igft(s_hat)
        >>> np.all((s - s_star) < 1e-10)
        True

        """
        if s.shape[0] != self.N:
            raise ValueError('First dimension should be the number of nodes '
                             'G.N = {}, got {}.'.format(self.N, s.shape))
        U = np.conjugate(self.U)  # True Hermitian. (Although U is often real.)
        return np.tensordot(U, s, ([0], [0]))

    def igft(self, s_hat):
        r"""Compute the inverse graph Fourier transform.

        The inverse graph Fourier transform of a Fourier domain signal
        :math:`\hat{s}` is defined as

        .. math:: s = U \hat{s},

        where :math:`U` is the Fourier basis :attr:`U`.

        Parameters
        ----------
        s_hat : ndarray
            Graph signal in the Fourier domain.

        Returns
        -------
        s : ndarray
            Representation of s_hat in the vertex domain.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.compute_fourier_basis()
        >>> s_hat = np.random.normal(size=(G.N, 5, 1))
        >>> s = G.igft(s_hat)
        >>> s_hat_star = G.gft(s)
        >>> np.all((s_hat - s_hat_star) < 1e-10)
        True

        """
        if s_hat.shape[0] != self.N:
            raise ValueError('First dimension should be the number of nodes '
                             'G.N = {}, got {}.'.format(self.N, s_hat.shape))
        return np.tensordot(self.U, s_hat, ([1], [0]))
