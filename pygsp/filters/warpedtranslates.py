# -*- coding: utf-8 -*-

from numpy import log, spacing
from . import Filter, HalfCosine, Itersine
from pygsp import gutils


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
            wf = lambda s: log(1 + wf(s) /
                                  xmax * lmax * logmax + spacing(1))

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

    def _pwl_warp_fn(self, x, y, s):
        pass

    def _mono_cubic_warp_fn(self, x, y, s):
        pass
