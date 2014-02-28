#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from pygsp import operators


class FilteringMethod:
    """Enum for different method of filtering"""
    CHEBY = 0
    SPECTRAL = 1


class GraphFilter(object):
    """Generic graph filter and approximation by Chebyshev polynomial."""
    def __init__(self, kernel):
        """Graph filter constructor

        Parameters
            ----------
            kernel : lambda filter function defined on [0, inf)
        """
        self.kernel = kernel
        self.description = "Generic graph filter"
        self.cheby_coeffs = None

    def __str__(self):
        return self.description

    def apply(self, g, signal, method):
        # if you don't have cheby coeff, compute them first
        if method == 0:
            return self._apply_cheby(g,signal)  
        elif method == 1:
            return self._apply_fourier(g,signal)  
        else:
            raise "Unknown computation methods"
        

    def _apply_cheby(self,g,signal):
        # TODO
        return 0

    def _apply_fourier(self,g,signal):
        hats = operators.gft(g,signal)
        hat = self.kernel(g.E) * hats
        return operators.igft(g, hat)


class HeatGraphFilter(GraphFilter):
    def __init__(self, tau):
        kernel = lambda x: np.exp(- tau * x)  # Heat kernel function
        super(HeatGraphFilter, self).__init__(kernel)
        self.description = "Heat kernel filter with tau: " + str(tau)

    def __str__(self):
        return self.description + "\n    lambda x: np.exp(- tau * x)"

class GaussianGraphFilter(GraphFilter):
    def __init__(self, tau):
        kernel = lambda x: np.exp(- x**2/tau)  # Heat kernel function
        super(HeatGraphFilter, self).__init__(kernel)
        self.description = "Gaussian filter with tau: " + str(tau)

    def __str__(self):
        return self.description + "\n    lambda x: np.exp(- x**2/tau)"

class RectGraphFilter(GraphFilter):
    def __init__(self, tau):
        kernel = lambda x: (x<tau)*1.0  # Heat kernel function
        super(HeatGraphFilter, self).__init__(kernel)
        self.description = "Rectangle kernel filter with tau: " + str(tau)

    def __str__(self):
        return self.description + "\n    lambda x: (x<tau)*1.0"


def compute_cheby_coeff(kernel, order, m, arange=(-1.0, 1.0)):
    """ Compute Chebyshev coefficients of given function.

        Parameters
        ----------
        g : function handle, should define function on arange
        order : maximum order Chebyshev coefficient to compute
        m : grid order used to compute quadrature (default is order+1)
        arange : interval of approximation (defaults to (-1,1) )

        Returns
        -------
        c : list of Chebyshev coefficients, ordered such that c(j+1) is
        j'th Chebyshev coefficient

    """

    if m < 1:
        m = order+1

    # Integral bounds    
    amin = (arange[1] - arange[0]) / 2.0
    amax = (arange[1] + arange[0]) / 2.0
    # Integral bounds    
    a = np.pi * (np.r_[1:m+1] - 0.5) / m
    s = kernel(a1 * np.cos(a) + a2)
    c = np.zeros(order+1)
    for j in range(order+1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c



def cheby_op(f, L, c, arange):  # copy/paste of Weinstein,must be adapted
    """Compute (possibly multiple) polynomials of laplacian (in Chebyshev
        basis) applied to input.

        Coefficients for multiple polynomials may be passed as a lis. This
        is equivalent to setting
        r[0] = cheby_op(f, L, c[0], arange)
        r[1] = cheby_op(f, L, c[1], arange)
        ...

        but is more efficient as the Chebyshev polynomials of L applied to f
         can be
        computed once and shared.

        Parameters
        ----------
        f : input vector
        L : graph laplacian (should be sparse)
        c : Chebyshev coefficients. If c is a plain array, then they are
        coefficients for a single polynomial. If c is a list, then it contains
        coefficients for multiple polynomials, such  that c[j](1+k) is k'th
        Chebyshev coefficient the j'th polynomial.
        arange : interval of approximation

        Returns
        -------
        r : If c is a list, r will be a list of vectors of size of f. If c is
        a plain array, r will be a vector the size of f.
        """
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op(f, L, [c], arange)
        return r[0]

    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0

    Twf_old = f
    Twf_cur = (L*f - a2*f) / a1
    r = [0.5*c[j][0]*Twf_old + c[j][1]*Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        Twf_new = (2/a1) * (L*Twf_cur - a2*Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k+1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r

# def view_filter(g, c_coeffs, lambdamax): #adapted from Weinstein view_design
#     """Plot the filter in the spectral domain and its Chebychev approx.

#         Plot the input scaling function and wavelet kernels, indicates the wavelet
#         scales by legend, and also shows the sum of squares G and corresponding
#         frame bounds for the transform.

#         Parameters
#         ----------
#         g : handle for the filter function
#         c_coeffs : Chebychev coefficients of the approximation of g
#         lambdamax : max spectral value, the curves are plotted on [0,lambdamax]

#         Returns
#         -------
#         h : figure handle
#         """
#     x = np.linspace(0, lambdamax, 1e3)
#     h = plt.figure()


#     plt.plot(x, g(x), 'k', label='g')
#     max_order =c_coeffs.size

#     a1 = lambdamax / 2.0
#     a2 = lambdamax / 2.0

#     f=1
#     Twf_old = f
#     Twf_cur = (x*f - a2*f) / a1
#     r = [0.5*c_coeffs[0]*Twf_old + c_coeffs[1]*Twf_cur]

#     for k in range(1, max_order-1):
#         Twf_new = (2/a1) * (x*Twf_cur - a2*Twf_cur) - Twf_old
#         r= r + c_coeffs[k+1] * Twf_new
#         Twf_old = Twf_cur
#         Twf_cur = Twf_new

#     plt.plot(x,r.T ,label='g_approx')
#     plt.xlim(0, lambdamax)

#     plt.title('filter kernel g in the spectral domain \n'
#               'and its Chebychev approximation')
#     plt.yticks(np.r_[0:4])
#     plt.ylim(0, 3)
#     plt.legend()
#     plt.show()
#     return h

# def apply_fiter(graph_props, filter, s):
#     """ Apply arbitrary filter to a signal residing on nodes of a graph.

#     Inputs:

#         - graph_props is the graph properties (object of SpectralProp class)
#         - filter: object of Filter class. It is applied in the graph
#         frequency domain of the specified graph type.
#         - s is the signal we want to filter

#     Output:

#         - Filtered signal
#     """
#     pass
