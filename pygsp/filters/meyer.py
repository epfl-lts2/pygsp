# -*- coding: utf-8 -*-

import numpy as np

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


_logger = utils.build_logger(__name__)


class Meyer(Filter):
    r"""
    Meyer filterbank

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters from 0 to lmax (default = 6).
    scales : ndarray
        Vector of scales to be used (default: log scale).

    References
    ----------
    Use of this kernel for SGWT proposed by Nora Leonardi and Dimitri Van De
    Ville in :cite:`leonardi2011wavelet`.

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Meyer(G)

    """

    def __init__(self, G, Nf=6, scales=None, **kwargs):

        if scales is None:
            scales = (4./(3 * G.lmax)) * np.power(2., np.arange(Nf-2, -1, -1))

        if len(scales) != Nf - 1:
            raise ValueError('The number of scales should be equal to '
                             'the number of filters minus 1.')

        g = [lambda x: kernel_meyer(scales[0] * x, 'scaling_function')]
        for i in range(Nf - 1):
            g.append(lambda x: kernel_meyer(scales[i] * x, 'wavelet'))

        def kernel_meyer(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asarray(x)

            l1 = 2/3.
            l2 = 4/3.  # 2*l1
            l3 = 8/3.  # 4*l1

            def v(x):
                return x**4 * (35 - 84*x + 70*x**2 - 20*x**3)

            r1ind = (x < l1)
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == 'scaling_function':
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
            elif kernel_type == 'wavelet':
                r[r2ind] = np.sin((np.pi/2) * v(np.abs(x[r2ind])/l1 - 1))
                r[r3ind] = np.cos((np.pi/2) * v(np.abs(x[r3ind])/l2 - 1))
            else:
                raise ValueError('Unknown kernel type {}'.format(kernel_type))

            return r

        super(Meyer, self).__init__(G, g, **kwargs)
