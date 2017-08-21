# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils


logger = utils.build_logger(__name__)


def gft(G, s):
    r"""
    Compute graph Fourier transform.

    The graph Fourier transform of a signal :math:`s` is defined as

    .. math:: \hat{s} = U^* s,

    where :math:`U` is the Fourier basis :py:attr:`pygsp.graphs.Graph.U` and
    :math:`U^*` denotes the conjugate transpose or Hermitian transpose of
    :math:`U`.

    Parameters
    ----------
    G : Graph or Fourier basis
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
    >>> s_hat = operators.gft(G, s)
    >>> s_star = operators.igft(G, s_hat)
    >>> np.linalg.norm(s - s_star) < 1e-10
    True

    """

    try:
        U = G.U
    except AttributeError:
        U = G

    return np.dot(np.conjugate(U.T), s)  # True Hermitian here.


def igft(G, s_hat):
    r"""
    Compute inverse graph Fourier transform.

    The inverse graph Fourier transform of a Fourier domain signal
    :math:`\hat{s}` is defined as

    .. math:: s = U \hat{s},

    where :math:`U` is the Fourier basis :py:attr:`pygsp.graphs.Graph.U`.

    Parameters
    ----------
    G : Graph or Fourier basis
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
    >>> s = operators.igft(G, s_hat)
    >>> s_hat_star = operators.gft(G, s)
    >>> np.linalg.norm(s_hat - s_hat_star) < 1e-10
    True

    """

    try:
        U = G.U
    except AttributeError:
        U = G

    return np.dot(U, s_hat)


def gft_windowed(G, g, f, lowmemory=True):
    r"""
    Windowed graph Fourier transform.

    Parameters
    ----------
    G : Graph
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

    Nf = np.shape(f)[1]

    if isinstance(g, list):
        g = igft(G, g[0](G.e))
    elif hasattr(g, '__call__'):
        g = igft(G, g(G.e))

    if not lowmemory:
        # Compute the Frame into a big matrix
        Frame = _gwft_frame_matrix(G, g)

        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N, Nf), order='F')

    else:
        # Compute the translate of g
        # TODO: use operators.translate()
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N) * np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))
        C = np.empty((G.N, G.N))

        for j in range(Nf):
            for i in range(G.N):
                C[:, i, j] = (np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.dot(np.kron(np.ones((G.N)), Ftrans[:, i])).T, f[:, j])

    return C


def gft_windowed_gabor(G, f, k):
    r"""
    Gabor windowed graph Fourier transform.

    Parameters
    ----------
    G : Graph
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
    >>> C = operators.gft_windowed_gabor(G, s, lambda x: x/(1.-x))
    >>> C.shape == (G.N, G.N)
    True

    """

    from pygsp import filters

    g = filters.Gabor(G, k)

    C = g.analysis(f)
    C = utils.vec2mat(C, G.N).T

    return C


def _gwft_frame_matrix(G, g):
    r"""
    Create the GWFT frame.

    Parameters
    ----------
    G : Graph
    g : window

    Returns
    -------
    F : ndarray
        Frame
    """

    if G.N > 256:
        logger.warning("It will create a big matrix. You can use other methods.")

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))
    F = utils.repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    return F


def gft_windowed_normalized(G, g, f, lowmemory=True):
    r"""
    Normalized windowed graph Fourier transform.

    Parameters
    ----------
    G : Graph
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

    if lowmemory:
        # Compute the Frame into a big matrix
        Frame = _ngwft_frame_matrix(G, g)
        C = np.dot(Frame.T, f)
        C = np.reshape(C, (G.N, G.N), order='F')

    else:
        # Compute the translate of g
        # TODO: use operators.translate()
        ghat = np.dot(G.U.T, g)
        Ftrans = np.sqrt(G.N)*np.dot(G.U, (np.kron(np.ones((1, G.N)), ghat)*G.U.T))
        C = np.empty((G.N, G.N))

        for i in range(G.N):
            atoms = np.kron(np.ones((G.N)), 1./G.U[:, 0])*G.U*np.kron(np.ones((G.N)), Ftrans[:, i]).T

            # normalization
            atoms /= np.kron((np.ones((G.N))), np.sqrt(np.sum(np.abs(atoms),
                                                              axis=0)))
            C[:, i] = np.dot(atoms, f)

    return C


def _ngwft_frame_matrix(G, g):
    r"""
    Create the NGWFT frame.

    Parameters
    ----------
    G : Graph
    g : ndarray
        Window

    Returns
    -------
    F : ndarray
        Frame

    """

    if G.N > 256:
        logger.warning('It will create a big matrix, you can use other methods.')

    ghat = np.dot(G.U.T, g)
    Ftrans = np.sqrt(g.N)*np.dot(G.U, (np.kron(np.ones((G.N)), ghat)*G.U.T))

    F = utils.repmatline(Ftrans, 1, G.N)*np.kron(np.ones((G.N)), np.kron(np.ones((G.N)), 1./G.U[:, 0]))

    # Normalization
    F /= np.kron((np.ones((G.N)), np.sqrt(np.sum(np.power(np.abs(F), 2),
                                          axiis=0))))

    return F
