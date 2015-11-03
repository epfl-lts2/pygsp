# -*- coding: utf-8 -*-
r"""
This module provides optimization tools to accelarate graph signal processing as a whole.
"""

from pygsp.data_handling import adj2vec
from pygsp.operators import operator
from pygsp.utils import build_logger

logger = build_logger(__name__)


def prox_tv(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, maxit=200, use_matrix=True):
    r"""
    Total Variation proximal operator for graphs.

    This function computes the TV proximal operator for graphs. The TV norm
    is the one norm of the gradient. The gradient is defined in the
    function :func:`~pygsp.operator.grad`.
    This function require the PyUNLocBoX to be executed.

    pygsp.optimization.prox_tv(y, gamma, param) solves:

    :math:`sol = \min_{z} \frac{1}{2} \|x - z\|_2^2 + \gamma  \|x\|_{TV}`

    Parameters
    ----------
    x: int
        Input signal
    gamma: ndarray
        Regularization parameter
    G: graph object
        Graphs structure
    A: lambda function
        Forward operator, this parameter allows to solve the following problem:
        :math:`sol = \min_{z} \frac{1}{2} \|x - z\|_2^2 + \gamma  \| A x\|_{TV}`
        (default = Id)
    At: lambda function
        Adjoint operator. (default = Id)
    nu: float
        Bound on the norm of the operator (default = 1)
    tol: float
        Stops criterion for the loop. The algorithm will stop if :
        :math:`\frac{n(t) - n(t - 1)} {n(t)} < tol`
        where :math: `n(t) = f(x) + 0.5 \|x-y\|_2^2` is the objective function at iteration :math:`t`
        (default = :math:`10e-4`)
    maxit: int
        Maximum iteration. (default = 200)
    use_matrix: bool
        If a matrix should be used. (default = True)

    Returns
    -------
    sol: solution

    Examples
    --------
    >>> from pygsp import optimization, graphs

    """
    if A is None:
        def A(x):
            return x
    if At is None:
        def At(x):
            return x

    if not hasattr(G, 'v_in'):
        adj2vec(G)

    tight = 0
    l1_nu = 2 * G.lmax * nu

    if use_matrix:
        def l1_a(x):
            return G.Diff * A(x)

        def l1_at(x):
            return G.Diff * At(D.T * x)
    else:
        def l1_a(x):
            return operator.grad(G, A(x))

        def l1_at(x):
            return operator.div(G, x)

    pyunlocbox.prox_l1(x, gamma, A=l1_a, At=l1_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
