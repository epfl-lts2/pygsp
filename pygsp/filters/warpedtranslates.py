# -*- coding: utf-8 -*-

import numpy as np
from . import Filter, HalfCosine, Itersine
from pygsp import gutils, utils

logger = utils.build_logger(__name__)


class WarpedTranslates(Filter):
    r"""
    Creates a vertex frequency filterbank

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters (default = #TODO)
    use_log : bool
        To add on the other warping a log function. This is an alternative
        way of constructing a spectral graph wavelets. These are adapted to
        the specific spectrum not only it's lenght.
        The final warping function will look like.

        .. math:: /log(f(x))

        (Default is False)

    logmax : float
        Maximum of the log function?

    base_filter = str
        The name of the initial uniform filterbank. 'itersine' and 'half_cosine'
        are the only two usable values.

    warping_type : str
        Creates a warping function according to three different methods :
        - 'spectrum_approximation'
        - 'spectrum_interpolation'
        - 'custom'
        More informations in the Note section
        (Default is spectrum_approximation)

    warp_function : lambda
        To be used with the 'custom' warping_type, it provides a way to use
        homemade warping_function.
        (Default is None)



    Returns
    -------
    out : WarpedTranslates

    Examples
    --------
    Not Implemented for now
    # >>> from pygsp import graphs, filters
    # >>> G = graphs.Logo()
    # >>> F = filters.WarpedTranslates(G)

    See :cite:`shuman2013spectrum`

    """

    def __init__(self, G, Nf=6, use_log=False, logmax=10, base_filter=None,
                 overlap=2, interpolation_type='monocubic',
                 warping_type='spectrum_approximation',
                 approx_spectrum=[0, 0],
                 **kwargs):

        super(WarpedTranslates, self).__init__(G, **kwargs)

        # Check estimate lmax
        # Nocheck
        lmax = gutils.estimate_lmax(G)

        # Choose good case folloeing the warping type:
        # Can be : 'spectrum_interpolation', 'spectrum_approximation', 'custom'
        if warping_type == 'spectrum_interpolation':
            if interpolation_type == 'pwl':
                wf = lambda s: self._pwl_warp_fn(approx_spectrum[0],
                                                 approx_spectrum[1], s)
            elif interpolation_type == 'monocubic':
                wf = lambda s: self._mono_cubic_warp_fn(approx_spectrum[0],
                                                        approx_spectrum[1], s)

        elif warping_type == 'spectrum_interpolation':
            pass

        elif warping_type == 'custom':
            pass

        # if log calculate recalculate the warped fucntion
        if use_log:
            xmax = wf(lmax)
            wf = lambda s: np.log(1 + wf(s) /
                               xmax * lmax * logmax + np.spacing(1))

        # We have to generate uniform translates covering for half_cosine or itersine
        if isinstance(base_filter, list):
            uniform_filters = []
            for i in range(Nf):
                uniform_filters.append(lambda x, ind=i:
                                       base_filter[i](x * lmax / wf(lmax)))
        else:
            if base_filter == 'half_cosine':
                uniform_filters = HalfCosine(wf(lmax), Nf)
            elif base_filter == 'itersine':
                uniform_filters = Itersine(wf(lmax), Nf, overlap)

        return uniform_filters

    def _pwl_warp_fn(self, x, y, x0):
        cut = 1e-4
        if np.max(x0) > np.max(x) + cut or min(x0) < min(x) - cut:
            logger.error('This function does not allow you to interpolate outside x and y')

        mat_x_y = []
        for ele, ind in enumerate(x):
            mat_x_y.append((x[ind], y[ind]))

        for ele  in mat_x_y:
            sorted_x_y = np.sort(mat_x_y[:][0])
            x = sorted_x_y[:][0]
            y = sorted_x_y[:][1]

        # To make sure the data is sorted and monotonic
        if np.sort(x) == x and np.sort(y) == y:
            logger.error('Data points are not monotonic.')

        n_pts = x.size
        inter_val = np.zeros(x0.size)

        for i in range(x0.size):
            close_ind = np.min(np.abs(x - x0[i]))
            if x[close_ind] - x0[i] < (-cut) or np.abs(x[close_ind] - x0[i] < cut and close_ind < n_pts):
                low_ind = close_ind
            else:
                low_ind = close_ind - 1

            inter_val[i] = y[low_ind] * (x[low_ind + 1] - x0[i])/(x[low_ind + 1] - x[low_ind]) + y[low_ind + 1] * (x0[i] - x[low_ind])/(x[low_ind + 1] - x[low_ind])

            return inter_val.reshape(inter_val, x0.size)


    def _mono_cubic_warp_fn(self, x, y, x0):
        r"""
        Returns interpolated values
        """
        cut = 1e-4

        # Monotonic cubix interpolation using the Fritsch-Carlson method
        if not x.size == y.size:
            logger.error('x and y vectors have to be the same size.')

        # To make sure the data is sorted and monotonic
        if np.sort(x) == x and np.sort(y) == y:
            logger.error('Data points are not monotonic.')

        # Compute the slopes of secant lines
        n_pts = x.size
        Delta = (y[1:] - y[:n_pts])/(x[1:] - x[:n_pts])

        # Initialize tangents m at every data points
        m = (Delta[:n_pts - 1] + Delta[1:n_pts])/2
        # To mitigate side effects
        m = Delta[0].append(m).append(Delta[-1])

        # Check for equal y's to set slope equal to zero
        for k, i in enumerate(Delta):
            if k == 0:
                m[i] = 0
                m[i + 1] = 0

        # alpha and beta initialization
        alpha = m[:n_pts]/Delta
        beta = m[1:-1]/Delta

        # Make monotonic
        for k in range(n_pts):
            if alpha[k] ** 2 + beta[k] ** 2 > 9:
                tau = 3/np.sqrt((alpha[k]**2 + beta[k]**2))
                m[k] = tau * alpha[k] * Delta[k]
                m[k + 1] = tau * beta[k] * Delta[k]

        # Cubic interpolation
        n_pts_inter = x0.size
        inter_val =  np.zeros(x0.shape)

        for i in range(n_pts_inter):
            close_ind = np.min(np.abs(x - x0[i]))
            if x[close_ind] - x0[i] < -cut or np.abs(x[close_ind] - x0[i] < cut) and close_ind < n_pts:
                low_ind = close_ind - 1
            else:
                low_ind = close_ind - 1
            h = x[low_ind + 1] - x[low_ind]
            t = (x[i] - x[low_ind]) / h

            inter_val[i] = y[low_ind] * (2 * t ** 3 - 3 * t ** 2 + 1)\
                + h * m[low_ind] * (t ** 3 - 2 * t ** 2 + t)\
                + h * m[low_ind] * (t ** 3 - t ** 2)

        return inter_val
