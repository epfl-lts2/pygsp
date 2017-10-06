# -*- coding: utf-8 -*-

import numpy as np

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

        Is computed by :func:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('U', 'Fourier basis')

    @property
    def e(self):
        r"""Eigenvalues of the Laplacian (square of graph frequencies).

        Is computed by :func:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('e', 'eigenvalues vector')

    @property
    def mu(self):
        r"""Coherence of the Fourier basis.

        Is computed by :func:`compute_fourier_basis`.
        """
        return self._check_fourier_properties('mu', 'Fourier basis coherence')

    def compute_fourier_basis(self, recompute=False):
        r"""Compute the Fourier basis of the graph (cached).

        The result is cached and accessible by the :attr:`U`, :attr:`e`,
        :attr:`lmax`, and :attr:`mu` properties.

        Parameters
        ----------
        recompute: bool
            Force to recompute the Fourier basis if already existing.

        Notes
        -----
        'G.compute_fourier_basis()' computes a full eigendecomposition of
        the graph Laplacian :math:`L` such that:

        .. math:: L = U \Lambda U^*,

        where :math:`\Lambda` is a diagonal matrix of eigenvalues and the
        columns of :math:`U` are the eigenvectors.

        *G.e* is a vector of length *G.N* containing the Laplacian
        eigenvalues. The largest eigenvalue is stored in *G.lmax*.
        The eigenvectors are stored as column vectors of *G.U* in the same
        order that the eigenvalues. Finally, the coherence of the
        Fourier basis is found in *G.mu*.

        References
        ----------
        See :cite:`chung1997spectral`.

        Examples
        --------
        >>> G = graphs.Torus()
        >>> G.compute_fourier_basis()
        >>> G.U.shape
        (256, 256)
        >>> G.e.shape
        (256,)
        >>> G.lmax == G.e[-1]
        True
        >>> G.mu < 1
        True

        """

        if hasattr(self, '_e') and hasattr(self, '_U') and not recompute:
            return

        assert self.L.shape == (self.N, self.N)
        if self.N > 3000:
            self.logger.warning('Computing the full eigendecomposition of a '
                                'large matrix ({0} x {0}) may take some '
                                'time.'.format(self.N))

        # TODO: handle non-symmetric Laplatians. Test lap_type?

        self._e, self._U = np.linalg.eigh(self.L.toarray())
        # Columns are eigenvectors. Sorted in ascending eigenvalue order.

        # Smallest eigenvalue should be zero: correct numerical errors.
        # Eigensolver might sometimes return small negative values, which
        # filter's implementations may not anticipate. Better for plotting too.
        assert -1e-12 < self._e[0] < 1e-12
        self._e[0] = 0

        if self.lap_type == 'normalized':
            # Spectrum bounded by [0, 2].
            assert self._e[-1] <= 2

        assert np.max(self._e) == self._e[-1]
        self._lmax = self._e[-1]
        self._mu = np.max(np.abs(self._U))

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

    def translate(self, f, i):
        r"""Translate the signal *f* to the node *i*.

        Parameters
        ----------
        f : ndarray
            Signal
        i : int
            Indices of vertex

        Returns
        -------
        ft : translate signal

        """

        raise NotImplementedError('Current implementation is not working.')

        fhat = self.gft(f)
        nt = np.shape(f)[1]

        ft = self.igft(fhat, np.kron(np.ones((1, nt)), self.U[i]))
        ft *= np.sqrt(self.N)

        return ft

    def gft_windowed_gabor(self, s, k):
        r"""Gabor windowed graph Fourier transform.

        Parameters
        ----------
        s : ndarray
            Graph signal in the vertex domain.
        k : function
            Gabor kernel. See :class:`pygsp.filters.Gabor`.

        Returns
        -------
        s : ndarray
            Vertex-frequency representation of the signals.

        Examples
        --------
        >>> G = graphs.Logo()
        >>> s = np.random.normal(size=(G.N, 2))
        >>> s = G.gft_windowed_gabor(s, lambda x: x/(1.-x))
        >>> s.shape
        (1130, 2, 1130)

        """
        from pygsp import filters
        return filters.Gabor(self, k).filter(s)

    def gft_windowed(self, g, f, lowmemory=True):
        r"""Windowed graph Fourier transform.

        Parameters
        ----------
        g : ndarray or Filter
            Window (graph signal or kernel).
        f : ndarray
            Graph signal in the vertex domain.
        lowmemory : bool
            Use less memory (default=True).

        Returns
        -------
        C : ndarray
            Coefficients.

        """

        raise NotImplementedError('Current implementation is not working.')

        N = self.N
        Nf = np.shape(f)[1]
        U = self.U

        if isinstance(g, list):
            g = self.igft(g[0](self.e))
        elif hasattr(g, '__call__'):
            g = self.igft(g(self.e))

        if not lowmemory:
            # Compute the Frame into a big matrix
            Frame = self._frame_matrix(g, normalize=False)

            C = np.dot(Frame.T, f)
            C = np.reshape(C, (N, N, Nf), order='F')

        else:
            # Compute the translate of g
            # TODO: use self.translate()
            ghat = np.dot(U.T, g)
            Ftrans = np.sqrt(N) * np.dot(U, (np.kron(np.ones((N)), ghat)*U.T))
            C = np.empty((N, N))

            for j in range(Nf):
                for i in range(N):
                    C[:, i, j] = (np.kron(np.ones((N)), 1./U[:, 0])*U*np.dot(np.kron(np.ones((N)), Ftrans[:, i])).T, f[:, j])

        return C

    def gft_windowed_normalized(self, g, f, lowmemory=True):
        r"""Normalized windowed graph Fourier transform.

        Parameters
        ----------
        g : ndarray
            Window.
        f : ndarray
            Graph signal in the vertex domain.
        lowmemory : bool
            Use less memory. (default = True)

        Returns
        -------
        C : ndarray
            Coefficients.

        """

        raise NotImplementedError('Current implementation is not working.')

        N = self.N
        U = self.U

        if lowmemory:
            # Compute the Frame into a big matrix
            Frame = self._frame_matrix(g, normalize=True)
            C = np.dot(Frame.T, f)
            C = np.reshape(C, (N, N), order='F')

        else:
            # Compute the translate of g
            # TODO: use self.translate()
            ghat = np.dot(U.T, g)
            Ftrans = np.sqrt(N)*np.dot(U, (np.kron(np.ones((1, N)), ghat)*U.T))
            C = np.empty((N, N))

            for i in range(N):
                atoms = np.kron(np.ones((N)), 1./U[:, 0])*U*np.kron(np.ones((N)), Ftrans[:, i]).T

                # normalization
                atoms /= np.kron((np.ones((N))), np.sqrt(np.sum(np.abs(atoms),
                                                                  axis=0)))
                C[:, i] = np.dot(atoms, f)

        return C

    def _frame_matrix(self, g, normalize=False):
        r"""Create the GWFT frame.

        Parameters
        ----------
        g : window

        Returns
        -------
        F : ndarray
            Frame
        """

        N = self.N
        U = self.U

        if self.N > 256:
            logger.warning("It will create a big matrix. You can use other methods.")

        ghat = np.dot(U.T, g)
        Ftrans = np.sqrt(N)*np.dot(U, (np.kron(np.ones(N), ghat)*U.T))
        F = utils.repmatline(Ftrans, 1, N)*np.kron(np.ones((N)), np.kron(np.ones((N)), 1./U[:, 0]))

        if normalize:
            F /= np.kron((np.ones(N), np.sqrt(np.sum(np.power(np.abs(F), 2), axis=0))))

        return F
