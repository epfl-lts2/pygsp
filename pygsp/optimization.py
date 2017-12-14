# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.optimization` module provides tools for convex optimization on
graphs.
"""

from pygsp import utils


logger = utils.build_logger(__name__)


def _import_pyunlocbox():
    try:
        from pyunlocbox import functions, solvers
    except Exception:
        raise ImportError('Cannot import pyunlocbox, which is needed to solve '
                          'this optimization problem. Try to install it with '
                          'pip (or conda) install pyunlocbox.')
    return functions, solvers


def prox_tv(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, maxit=200, use_matrix=True):
    r"""
    Total Variation proximal operator for graphs.

    This function computes the TV proximal operator for graphs. The TV norm
    is the one norm of the gradient. The gradient is defined in the
    function :meth:`pygsp.graphs.Graph.grad`.
    This function requires the PyUNLocBoX to be executed.

    This function solves:

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
        where :math:`n(t) = f(x) + 0.5 \|x-y\|_2^2` is the objective function at iteration :math:`t`
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

    """
    if A is None:
        def A(x):
            return x
    if At is None:
        def At(x):
            return x

    tight = 0
    l1_nu = 2 * G.lmax * nu

    if use_matrix:
        def l1_a(x):
            return G.Diff * A(x)

        def l1_at(x):
            return G.Diff * At(D.T * x)
    else:
        def l1_a(x):
            return G.grad(A(x))

        def l1_at(x):
            return G.div(x)

    functions, _ = _import_pyunlocbox()
    functions.norm_l1(x, gamma, A=l1_a, At=l1_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
