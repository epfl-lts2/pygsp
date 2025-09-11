import numpy as np

from pygsp import utils

from .filter import Filter  # prevent circular import in Python < 3.5


class Meyer(Filter):
    r"""Design a filter bank of Meyer wavelets (tight frame).

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

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Meyer(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = g.plot(ax=axes[0])
    >>> _ = G.plot(s, ax=axes[1])

    """

    def __init__(self, G, Nf=6, scales=None):
        if scales is None:
            scales = (4.0 / (3 * G.lmax)) * np.power(2.0, np.arange(Nf - 2, -1, -1))
        self.scales = scales

        if len(scales) != Nf - 1:
            raise ValueError("len(scales) should be Nf-1.")

        kernels = [lambda x: kernel(scales[0] * x, "scaling_function")]

        for i in range(Nf - 1):
            kernels.append(lambda x, i=i: kernel(scales[i] * x, "wavelet"))

        def kernel(x, kernel_type):
            r"""
            Evaluates Meyer function and scaling function

            * meyer wavelet kernel: supported on [2/3,8/3]
            * meyer scaling function kernel: supported on [0,4/3]
            """

            x = np.asanyarray(x)

            l1 = 2 / 3.0
            l2 = 4 / 3.0  # 2*l1
            l3 = 8 / 3.0  # 4*l1

            def v(x):
                return x**4 * (35 - 84 * x + 70 * x**2 - 20 * x**3)

            r1ind = x < l1
            r2ind = (x >= l1) * (x < l2)
            r3ind = (x >= l2) * (x < l3)

            # as we initialize r with zero, computed function will implicitly
            # be zero for all x not in one of the three regions defined above
            r = np.zeros(x.shape)
            if kernel_type == "scaling_function":
                r[r1ind] = 1
                r[r2ind] = np.cos((np.pi / 2) * v(np.abs(x[r2ind]) / l1 - 1))
            elif kernel_type == "wavelet":
                r[r2ind] = np.sin((np.pi / 2) * v(np.abs(x[r2ind]) / l1 - 1))
                r[r3ind] = np.cos((np.pi / 2) * v(np.abs(x[r3ind]) / l2 - 1))
            else:
                raise ValueError(f"Unknown kernel type {kernel_type}")

            return r

        super().__init__(G, kernels)
