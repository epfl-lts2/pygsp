# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.learning` module implements functions to solve learning
problems.

Semi-supervized learning
========================

Those functions help to solve a semi-supervized learning problem, i.e. a
problem where only some values of a graph signal are known and the others shall
be inferred.

.. autosummary::

    regression_tik
    classification_tik

"""

import numpy as np
import scipy

from pyunlocbox import functions, solvers


def classification_tik_simplex(G, y, M, tau=0.1, **kwargs):
    r"""Solve a classification problem on graph via Tikhonov minimization 
    with simple constraints.

    The function first transform :math:`y` in logits :math:`Y`. It then solves

    .. math:: \operatorname*{arg min}_X \| M X - Y \|_2^2 + \tau \ tr(X^T L X) 
                                        s.t. sum(Y) = 1 and Y>=0

    otherwise, where :math:`X` and :math:`Y` are logits. The function returns
    the logits.

    Parameters
    ----------
    G : Graph
    y : array of length G.N
        Measurements
    M : array of boolean, length G.N
        Masking vector.
    tau : float
        Regularization parameter.

    Examples
    --------
    >>> from pygsp import graphs, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Logo()
    >>> G.estimate_lmax()

    Create a ground truth signal:

    >>> signal = np.zeros(G.N)
    >>> signal[G.info['idx_s']] = 1
    >>> signal[G.info['idx_p']] = 2

    Construct a measurements signal from a binary mask:

    >>> rs = np.random.RandomState(42)
    >>> mask = rs.uniform(0, 1, G.N) > 0.5
    >>> measurements = signal.copy()
    >>> measurements[~mask] = np.nan

    Solve the classification problem by reconstructing the signal:

    >>> recovery = learning.classification_tik_simplex(G, measurements, mask, tau=0.1, verbosity='NONE')

    Plot the results. Note that recovery gives the logits, we recover the class
    using `np.argmax(recovery, axis=1)`

    >>> fig, ax = plt.subplots(2, 3, sharey=True, figsize=(10, 6))
    >>> (ax1, ax2, ax3), (ax4, ax5, ax6)  = ax
    >>> _ = G.plot_signal(signal, ax=ax1)
    >>> _ = ax1.set_title('Ground truth')
    >>> _ = G.plot_signal(measurements, ax=ax2)
    >>> _ = ax2.set_title('Measurements')
    >>> _ = G.plot_signal(np.argmax(recovery, axis=1), ax=ax3)
    >>> _ = ax3.set_title('Max logit')
    >>> _ = G.plot_signal(recovery[:,0], ax=ax4)
    >>> _ = ax4.set_title('Logit 0')
    >>> _ = G.plot_signal(recovery[:,1], ax=ax5)
    >>> _ = ax5.set_title('Logit 1')
    >>> _ = G.plot_signal(recovery[:,2], ax=ax6)
    >>> _ = ax6.set_title('Logit 2')
    >>> _ = fig.tight_layout()

    """
    assert(tau > 0)

    def to_logits(x):
        l = np.zeros([len(x), np.max(x)+1])
        l[range(len(x)), x] = 1
        return l
    y[M == False] = 0
    Y = to_logits(y.astype(np.int))
    Y[M == False, :] = 0

    def proj_simplex(y):
        d = y.shape[1]
        a = np.ones(d)
        idx = np.argsort(y)

        def evalpL(y, k, idx):
            return np.sum(y[idx[k:]] - y[idx[k]]) - 1

        def bisectsearch(idx, y):
            idxL, idxH = 0, d-1
            L = evalpL(y, idxL, idx)
            H = evalpL(y, idxH, idx)

            if L < 0:
                return idxL

            while (idxH-idxL) > 1:
                iMid = int((idxL+idxH)/2)
                M = evalpL(y, iMid, idx)

                if M > 0:
                    idxL, L = iMid, M
                else:
                    idxH, H = iMid, M

            return idxH

        def proj(idx, y):
            k = bisectsearch(idx, y)
            lam = (np.sum(y[idx[k:]])-1)/(d-k)
            return np.maximum(0, y-lam)
        x = np.zeros(y.shape)
        for i in range(len(y)):
            x[i] = proj(idx[i], y[i])
        # x = np.stack(map(proj,idx,y))

        return x

    f1 = functions.func()

    def smooth_eval(x):
        xTLx = np.sum(x * (G.L.dot(x)))
        e = M*((M*x.T)-Y.T)
        l2 = np.sum(e*e)
        return tau*xTLx + l2

    def smooth_grad(x):
        return 2*((M*(M*x.T-Y.T)).T + tau*G.L*x)
    f1._eval = smooth_eval
    f1._grad = smooth_grad

    f2 = functions.func()
    f2._eval = lambda x: 0        # Indicator functions evaluate to zero.

    def prox(x, step):
        return proj_simplex(x)
    f2._prox = prox

    step = 1/(2*(1+tau*G.lmax))
    solver = solvers.forward_backward(step=step)

    ret = solvers.solve([f1, f2], Y.copy(), solver, **kwargs)
    return ret['sol']


def classification_tik(G, y, M, tau=0):
    r"""Solve a classification problem on graph via Tikhonov minimization.

    The function first transform :math:`y` in logits :math:`Y`. It then solves

    .. math:: \operatorname*{arg min}_X \| M X - Y \|_2^2 + \tau \ tr(X^T L X)

    if :math:`\tau > 0` and

    .. math:: \operatorname*{arg min}_X tr(X^T L X) \ \text{ s. t. } \ Y = M X

    otherwise, where :math:`X` and :math:`Y` are logits. The function return
    the maximum of the logits.

    Parameters
    ----------
    G : Graph
    y : array of length G.N
        Measurements
    M : array of boolean, length G.N
        Masking vector.
    tau : float
        Regularization parameter.

    Examples
    --------
    >>> from pygsp import graphs, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Logo()

    Create a ground truth signal:

    >>> signal = np.zeros(G.N)
    >>> signal[G.info['idx_s']] = 1
    >>> signal[G.info['idx_p']] = 2

    Construct a measurements signal from a binary mask:

    >>> rs = np.random.RandomState(42)
    >>> mask = rs.uniform(0, 1, G.N) > 0.5
    >>> measurements = signal.copy()
    >>> measurements[~mask] = np.nan

    Solve the classification problem by reconstructing the signal:

    >>> recovery = learning.classification_tik(G, measurements, mask, tau=0)

    Plot the results:

    >>> fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    >>> _ = G.plot_signal(signal, ax=ax1)
    >>> _ = ax1.set_title('Ground truth')
    >>> _ = G.plot_signal(measurements, ax=ax2)
    >>> _ = ax2.set_title('Measurements')
    >>> _ = G.plot_signal(recovery, ax=ax3)
    >>> _ = ax3.set_title('Recovery')
    >>> _ = fig.tight_layout()

    """

    def to_logits(x):
        l = np.zeros([len(x), np.max(x)+1])
        l[range(len(x)), x] = 1
        return l
    y[M == False] = 0
    Y = to_logits(y.astype(np.int))
    X = regression_tik(G, Y, M, tau)

    return np.argmax(X, axis=1)


def regression_tik(G, y, M, tau=0):
    r"""Solve a regression problem on graph via Tikhonov minimization.

    If :math:`\tau > 0`:

    .. math:: \operatorname*{arg min}_x \| M x - y \|_2^2 + \tau \ x^T L x

    else:

    .. math:: \operatorname*{arg min}_x x^T L x \ \text{ s. t. } \ y = M x

    Parameters
    ----------
    G : Graph
    y : array of length G.N
        Measurements
    M : array of boolean, length G.N
        Masking vector.
    tau : float
        Regularization parameter.

    Examples
    --------
    >>> from pygsp import graphs, filters, learning
    >>> import matplotlib.pyplot as plt
    >>>
    >>> G = graphs.Sensor(N=100, seed=42)
    >>> G.estimate_lmax()

    Create a smooth ground truth signal:

    >>> filt = lambda x: 1 / (1 + 10*x)
    >>> g_filt = filters.Filter(G, filt)
    >>> rs = np.random.RandomState(42)
    >>> signal = g_filt.analyze(rs.randn(G.N))

    Construct a measurements signal from a binary mask:

    >>> mask = rs.uniform(0, 1, G.N) > 0.5
    >>> measurements = signal.copy()
    >>> measurements[~mask] = np.nan

    Solve the regression problem by reconstructing the signal:

    >>> recovery = learning.regression_tik(G, measurements, mask, tau=0)

    Plot the results:

    >>> f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    >>> c = [signal.min(), signal.max()]
    >>> _ = G.plot_signal(signal, ax=ax1, limits=c)
    >>> _ = ax1.set_title('Ground truth')
    >>> _ = G.plot_signal(measurements, ax=ax2, limits=c)
    >>> _ = ax2.set_title('Measurements')
    >>> _ = G.plot_signal(recovery, ax=ax3, limits=c)
    >>> _ = ax3.set_title('Recovery')
    >>> _ = fig.tight_layout()

    """

    if tau > 0:
        y[M == False] = 0
        # Creating this matrix may be problematic in term of memory.
        # Consider using an operator instead...
        if type(G.L).__module__ == np.__name__:
            LinearOp = np.diag(M*1) + tau * G.L
        else:
            def Op(x):
                return (M * x.T).T + tau * (G.L.dot(x))
            LinearOp = scipy.sparse.linalg.LinearOperator([G.N, G.N], Op)

        if type(G.L).__module__ == np.__name__:
            sol = np.linalg.solve(LinearOp, M * y)
        else:
            if len(y.shape) > 1:
                sol = np.zeros(shape=y.shape)
                res = np.zeros(shape=y.shape[1])
                for i in range(y.shape[1]):
                    sol[:, i], res[i] = scipy.sparse.linalg.cg(
                        LinearOp, y[:, i])
            else:
                sol, res = scipy.sparse.linalg.cg(LinearOp, y)
            # Do something with the residual...
        return sol

    else:

        if np.prod(M.shape) != G.N:
            raise ValueError("M should be of size [G.N,]")

        indl = M
        indu = M == False

        Luu = G.L[indu, :][:, indu]
        Wul = - G.L[indu, :][:, indl]
        if type(Luu).__module__ == np.__name__:
            sol_part = np.linalg.solve(Luu, np.matmul(Wul, y[indl]))
        else:
            sol_part = scipy.sparse.linalg.spsolve(Luu, Wul.dot(y[indl]))

        sol = y.copy()
        sol[indu] = sol_part

        return sol
