# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.learning` module implements some learning functions used
throughout the package.
"""

import numpy as np
import scipy


def classification_tik(G, y, M, tau=0):
    r"""Solve a semi-supervised classification problem on the graph.

    The function first transform y in logits Y. Then it solves

        argmin_X   || M X - Y ||_2^2 + tau tr(X^T L X)

    if tau > 0 and

        argmin_X   tr(X^T L X)   s. t.  Y = M X

    otherwise, where X and Y are logits. Eventually, the function return
    the maximum of the logits.

    Inputs
    ------
    G : Graph
    y : Measurements (numpy array [G.N, :])
    M : Mask (vector of boolean [G.N,])
    tau : regularization parameter

    Example
    -------
    import numpy as np
    import pygsp
    import matplotlib.pyplot as plt
    pygsp.plotting.BACKEND = 'matplotlib'

    G = pygsp.graphs.Logo()
    idx_g = np.squeeze(G.info['idx_g'])
    idx_s = np.squeeze(G.info['idx_s'])
    idx_p = np.squeeze(G.info['idx_p'])
    sig = np.zeros([G.N])
    sig[idx_s] = 1
    sig[idx_p] = 2

    # Make the input signal
    M = np.random.uniform(0,1,[G.N])>0.5 # Mask

    measurements = sig.copy()
    measurements[M==False] = np.nan

    # Solve the classification problem
    recovery = pygsp.learning.classification_tik(G, measurements, M, tau=0)

    # Plotting
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14,4))
    G.plot_signal(sig, plot_name = 'Ground truth', ax=ax1)
    G.plot_signal(measurements, plot_name = 'Measurement', ax=ax2)
    G.plot_signal(recovery, plot_name = 'Recovery', ax=ax3)
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
    r"""Solve a regression problem on the graph

    If tau > 0:

        argmin_x   || M x - y ||_2^2 + tau x^T L x

    else:

        argmin_x   x^T L x    s. t.  y = M x

    Inputs
    ------
    G : Graph
    y : Measurements (numpy array [G.N, :])
    M : Mask (vector of boolean [G.N,])
    tau : regularization parameter

    Examples
    --------
    import numpy as np
    import pygsp
    import matplotlib.pyplot as plt
    pygsp.plotting.BACKEND = 'matplotlib'

    # Create the graph
    G = pygsp.graphs.Sensor(N=100)
    G.estimate_lmax()

    # Create a smooth signal
    filt = lambda x: 1 / (1+10*x)
    g_filt = pygsp.filters.Filter(G,filt)
    sig = g_filt.analyze(np.random.randn(G.N,1))

    # Make the input signal
    M = np.random.uniform(0,1,[G.N])>0.5 # Mask

    measurements = sig.copy()
    measurements[M==False] = np.nan

    # Reconstructing the signal
    recovery = pygsp.learning.regression_tik(G, measurements, M, tau=0)

    # Plotting
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(9,3))
    c = [np.min(sig), np.max(sig[:,0])]
    G.plot_signal(sig, plot_name = 'Ground truth', ax=ax1, limits=c)
    G.plot_signal(measurements plot_name = 'Measurement', ax=ax2, limits=c)
    G.plot_signal(recovery, plot_name = 'Recovery', ax=ax3, limits=c)

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
            ValueError("M should be of size [G.N,]")

        indl = M
        indu = M == False

        Luu = G.L[indu, :][:, indu]
        Wul = - G.L[indu, :][:, indl]
        if type(Luu).__module__ == np.__name__:
            sol_part = np.linalg.solve(Luu, np.matmul(Wul,y[indl]))
        else:
            sol_part = scipy.sparse.linalg.spsolve(Luu, Wul.dot(y[indl]))

        sol = y.copy()
        sol[indu] = sol_part

        return sol
