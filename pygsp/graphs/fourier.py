# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils


logger = utils.build_logger(__name__)


class GraphFourier(object):

    def gft(self, s):
        r"""
        Compute graph Fourier transform.

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
        >>> import numpy as np
        >>> from pygsp import graphs, operators
        >>> G = graphs.Logo()
        >>> s = np.random.normal(size=G.N)
        >>> s_hat = G.gft(s)
        >>> s_star = G.igft(s_hat)
        >>> np.linalg.norm(s - s_star) < 1e-10
        True

        """
        return np.dot(np.conjugate(self.U.T), s)  # True Hermitian here.

    def igft(self, s_hat):
        r"""
        Compute inverse graph Fourier transform.

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
        >>> import numpy as np
        >>> from pygsp import graphs, operators
        >>> G = graphs.Logo()
        >>> s_hat = np.random.normal(size=G.N)
        >>> s = G.igft(s_hat)
        >>> s_hat_star = G.gft(s)
        >>> np.linalg.norm(s_hat - s_hat_star) < 1e-10
        True

        """
        return np.dot(self.U, s_hat)

    def gft_windowed_gabor(self, f, k):
        r"""
        Gabor windowed graph Fourier transform.

        Parameters
        ----------
        f : ndarray
            Graph signal in the vertex domain.
        k : function
            Gabor kernel. See :class:`pygsp.filters.Gabor`.

        Returns
        -------
        C : ndarray
            Coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from pygsp import graphs, operators
        >>> G = graphs.Logo()
        >>> s = np.random.normal(size=G.N)
        >>> C = G.gft_windowed_gabor(s, lambda x: x/(1.-x))
        >>> C.shape == (G.N, G.N)
        True

        """

        from pygsp import filters

        g = filters.Gabor(self, k)

        C = g.analysis(f)
        C = utils.vec2mat(C, self.N).T

        return C

    def gft_windowed(self, g, f, lowmemory=True):
        r"""
        Windowed graph Fourier transform.

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
            # TODO: use operators.translate()
            ghat = np.dot(U.T, g)
            Ftrans = np.sqrt(N) * np.dot(U, (np.kron(np.ones((N)), ghat)*U.T))
            C = np.empty((N, N))

            for j in range(Nf):
                for i in range(N):
                    C[:, i, j] = (np.kron(np.ones((N)), 1./U[:, 0])*U*np.dot(np.kron(np.ones((N)), Ftrans[:, i])).T, f[:, j])

        return C

    def gft_windowed_normalized(self, g, f, lowmemory=True):
        r"""
        Normalized windowed graph Fourier transform.

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
            # TODO: use operators.translate()
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
        r"""
        Create the GWFT frame.

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
