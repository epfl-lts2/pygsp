# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize

from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5


class Abspline(Filter):
    r"""Design an A B cubic spline wavelet filter bank.

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    scales : ndarray
        Vector of scales to be used.
        By default, initialized with :func:`pygsp.utils.compute_log_scales`.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Abspline(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, lpfactor=20, scales=None, **kwargs):

        def kernel_abspline3(x, alpha, beta, t1, t2):
            M = np.array([[1, t1, t1**2, t1**3],
                          [1, t2, t2**2, t2**3],
                          [0, 1, 2*t1, 3*t1**2],
                          [0, 1, 2*t2, 3*t2**2]])
            v = np.array([1, 1, t1**(-alpha) * alpha * t1**(alpha - 1),
                          -beta*t2**(- beta - 1) * t2**beta])
            a = np.linalg.solve(M, v)

            r1 = x <= t1
            r2 = (x >= t1)*(x < t2)
            r3 = (x >= t2)

            if isinstance(x, np.float64):

                if r1:
                    r = x[r1]**alpha * t1**(-alpha)
                if r2:
                    r = a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3
                if r3:
                    r = x[r3]**(-beta) * t2**beta

            else:
                r = np.zeros(x.shape)

                x2 = x[r2]

                r[r1] = x[r1]**alpha * t1**(-alpha)
                r[r2] = a[0] + a[1] * x2 + a[2] * x2**2 + a[3] * x2**3
                r[r3] = x[r3]**(-beta) * t2 ** beta

            return r

        G.lmin = G.lmax / lpfactor

        if scales is None:
            self.scales = utils.compute_log_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.scales = scales

        gb = lambda x: kernel_abspline3(x, 2, 2, 1, 2)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        g = [lambda x: 1.2 * np.exp(-1) * gl(x / lminfac)]
        for i in range(0, Nf - 1):
            g.append(lambda x, ind=i: gb(self.scales[ind] * x))

        f = lambda x: -gb(x)
        xstar = optimize.minimize_scalar(f, bounds=(1, 2),
                                         method='bounded')
        gamma_l = -f(xstar.x)
        lminfac = .6 * G.lmin
        g[0] = lambda x: gamma_l * gl(x / lminfac)

        super(Abspline, self).__init__(G, g, **kwargs)
