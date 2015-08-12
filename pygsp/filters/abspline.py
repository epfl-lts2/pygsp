# -*- coding: utf-8 -*-

from . import Filter

import numpy as np
from scipy import optimize
from math import exp


class Abspline(Filter):
    r"""
    Abspline Filterbank

    Inherits its methods from Filters

    Parameters
    ----------
    G : Graph
    Nf : int
        Number of filters from 0 to lmax (default = 6)
    lpfactor : int
        Low-pass factor lmin=lmax/lpfactor will be used to determine scales,
        the scaling function will be created to fill the lowpass gap.
        (default = 20)
    t : ndarray
        Vector of scale to be used (Initialized by default at
        the value of the log scale)

    Returns
    -------
    out : Abspline

    Examples
    --------
    >>> from pygsp import graphs, filters
    >>> G = graphs.Logo()
    >>> F = filters.Abspline(G)

    """

    def __init__(self, G, Nf=6, lpfactor=20, t=None, **kwargs):
        super(Abspline, self).__init__(G, **kwargs)

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

        if t is None:
            self.t = self.wlog_scales(G.lmin, G.lmax, Nf - 1)
        else:
            self.t = t

        gb = lambda x: kernel_abspline3(x, 2, 2, 1, 2)
        gl = lambda x: np.exp(-np.power(x, 4))

        lminfac = .4 * G.lmin

        self.g = [lambda x: 1.2 * exp(-1) * gl(x / lminfac)]
        for i in range(0, Nf - 1):
            self.g.append(lambda x, ind=i: gb(self.t[ind] * x))

        f = lambda x: -gb(x)
        xstar = optimize.minimize_scalar(f, bounds=(1, 2),
                                         method='bounded')
        gamma_l = -f(xstar.x)
        lminfac = .6 * G.lmin
        self.g[0] = lambda x: gamma_l * gl(x / lminfac)
