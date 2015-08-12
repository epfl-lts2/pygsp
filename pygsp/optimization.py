# -*- coding: utf-8 -*-

from pygsp.data_handling import adj2vec
from pygsp.operators import operator


def prox_tv(x, gamma, G, A=None, At=None, nu=1, tol=10e-4, verbose=1, maxit=200, use_matrix=True):
    r"""
    TV proximal operator for graphs.

    This function computes the TV proximal operator for graphs. The TV norm
    is the one norm of the gradient. The gradient is defined in the
    function |gsp_grad|. This function require the PyUNLocBoX to be executed.

    Parameters
    ----------
    x: int
        Description.
    gamma: array_like
        Description.
    G: graph object
        Description.
    A: lambda function
        Description.
    At: lambda function
        Description.
    nu: float
        Description.
    tol: float
        Description.
    verbose: int
        Description.
    maxit: int
        Description.
    use_matrix: bool
        Description.

    Returns
    -------
    sol: solution
        Description.

    Examples
    --------
    TODO

    """

    if A is None:
        A = lambda x: x
    if At is None:
        At = lambda x: x

    if not hasattr(G, 'v_in'):
        adj2vec(G)

    tight = 0
    l1_nu = 2 * G.lmax * nu

    if use_matrix:
        l1_a = lambda x: G.Diff * A(x)
        l1_at = lambda x: G.Diff * At(D.T * x)
    else:
        l1_a = lambda x: operator.grad(G, A(x))
        l1_at = lambda x: operator.div(G, x)

    pyunlocbox.prox_l1(x, gamma, A=l1_a, At=l1_at, tight=tight, maxit=maxit, verbose=verbose, tol=tol)
