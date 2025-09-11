r"""
The :mod:`pygsp.plotting` module implements functionality to plot PyGSP objects
with a `pyqtgraph <https://www.pyqtgraph.org>`_ or `matplotlib
<https://matplotlib.org>`_ drawing backend (which can be controlled by the
:data:`BACKEND` constant or individually for each plotting call).

Most users won't use this module directly.
Graphs (from :mod:`pygsp.graphs`) are to be plotted with
:meth:`pygsp.graphs.Graph.plot` and
:meth:`pygsp.graphs.Graph.plot_spectrogram`.
Filters (from :mod:`pygsp.filters`) are to be plotted with
:meth:`pygsp.filters.Filter.plot`.

.. data:: BACKEND

    The default drawing backend to use if none are provided to the plotting
    functions. Should be either ``'matplotlib'`` or ``'pyqtgraph'``. In general
    pyqtgraph is better for interactive exploration while matplotlib is better
    at generating figures to be included in papers or elsewhere.

"""

import functools

import numpy as np

from pygsp import utils

_logger = utils.build_logger(__name__)

BACKEND = "matplotlib"
_qtg_widgets = []
_plt_figures = []


def _import_plt():
    try:
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from mpl_toolkits import mplot3d
    except Exception as e:
        raise ImportError(
            "Cannot import matplotlib. Choose another backend "
            "or try to install it with "
            "pip (or conda) install matplotlib. "
            "Original exception: {}".format(e)
        )
    return mpl, plt, mplot3d


def _import_qtg():
    try:
        import pyqtgraph as qtg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtGui

        # Try to import QtWidgets for QApplication (needed for PyQt6)
        try:
            from pyqtgraph.Qt import QtWidgets
        except ImportError:
            QtWidgets = None

    except Exception as e:
        raise ImportError(
            "Cannot import pyqtgraph. Choose another backend "
            "or try to install it with "
            "pip (or conda) install pyqtgraph. You will also "
            "need PyQt6 (or PyQt5/PySide for older setups) and PyOpenGL. "
            "Original exception: {}".format(e)
        )
    return qtg, gl, QtGui, QtWidgets


def _get_qapplication(QtGui, QtWidgets):
    """Get QApplication from the appropriate Qt module."""
    # Try QtWidgets first (PyQt6), then fall back to QtGui (PyQt5)
    if QtWidgets and hasattr(QtWidgets, "QApplication"):
        return QtWidgets.QApplication
    elif hasattr(QtGui, "QApplication"):
        return QtGui.QApplication
    else:
        raise AttributeError("Cannot find QApplication in QtGui or QtWidgets")


def _plt_handle_figure(plot):
    r"""Handle the common work (creating an axis if not given, setting the
    title) of all matplotlib plot commands."""

    # Preserve documentation of plot.
    @functools.wraps(plot)
    def inner(obj, **kwargs):
        # Create a figure and an axis if none were passed.
        if kwargs["ax"] is None:
            _, plt, _ = _import_plt()
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)

            if (
                hasattr(obj, "coords")
                and obj.coords.ndim == 2
                and obj.coords.shape[1] == 3
            ):
                kwargs["ax"] = fig.add_subplot(111, projection="3d")
            else:
                kwargs["ax"] = fig.add_subplot(111)

        title = kwargs.pop("title")

        plot(obj, **kwargs)

        kwargs["ax"].set_title(title)

        try:
            fig.show(warn=False)
        except NameError:
            # No figure created, an axis was passed.
            pass

        return kwargs["ax"].figure, kwargs["ax"]

    return inner


def close_all():
    r"""Close all opened windows."""

    global _qtg_widgets
    for widget in _qtg_widgets:
        widget.close()
    _qtg_widgets = []

    global _plt_figures
    for fig in _plt_figures:
        _, plt, _ = _import_plt()
        plt.close(fig)
    _plt_figures = []


def show(*args, **kwargs):
    r"""Show created figures, alias to ``plt.show()``.

    By default, showing plots does not block the prompt.
    Calling this function will block execution.
    """
    _, plt, _ = _import_plt()
    plt.show(*args, **kwargs)


def close(*args, **kwargs):
    r"""Close last created figure, alias to ``plt.close()``."""
    _, plt, _ = _import_plt()
    plt.close(*args, **kwargs)


def _qtg_plot_graph(G, edges, vertex_size, title):
    qtg, gl, QtGui, QtWidgets = _import_qtg()

    if G.coords.shape[1] == 2:
        widget = qtg.GraphicsLayoutWidget()
        view = widget.addViewBox()
        view.setAspectLocked()

        if edges:
            pen = tuple(np.array(G.plotting["edge_color"]) * 255)
        else:
            pen = None

        adj = _get_coords(G, edge_list=True)

        g = qtg.GraphItem(pos=G.coords, adj=adj, pen=pen, size=vertex_size / 10)
        view.addItem(g)

    elif G.coords.shape[1] == 3:
        QApplication = _get_qapplication(QtGui, QtWidgets)
        if not QApplication.instance():
            QApplication([])  # We want only one application.
        widget = gl.GLViewWidget()
        widget.opts["distance"] = 10

        if edges:
            x, y, z = _get_coords(G)
            pos = np.stack((x, y, z), axis=1)
            g = gl.GLLinePlotItem(pos=pos, mode="lines", color=G.plotting["edge_color"])
            widget.addItem(g)

        gp = gl.GLScatterPlotItem(
            pos=G.coords, size=vertex_size / 3, color=G.plotting["vertex_color"]
        )
        widget.addItem(gp)

    widget.setWindowTitle(title)
    widget.show()

    global _qtg_widgets
    _qtg_widgets.append(widget)


def _plot_filter(filters, n, eigenvalues, sum, labels, title, ax, **kwargs):
    r"""Plot the spectral response of a filter bank.

    Parameters
    ----------
    n : int
        Number of points where the filters are evaluated.
    eigenvalues : boolean
        Whether to show the eigenvalues of the graph Laplacian.
        The eigenvalues should have been computed with
        :meth:`~pygsp.graphs.Graph.compute_fourier_basis`.
        By default, the eigenvalues are shown if they are available.
    sum : boolean
        Whether to plot the sum of the squared magnitudes of the filters.
        Default False if there is only one filter in the bank, True otherwise.
    labels : boolean
        Whether to label the filters.
        Default False if there is only one filter in the bank, True otherwise.
    title : str
        Title of the figure.
    ax : :class:`matplotlib.axes.Axes`
        Axes where to draw the graph. Optional, created if not passed.
    kwargs : dict
        Additional parameters passed to the matplotlib plot function.
        Useful for example to change the linewidth, linestyle, or set a label.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The figure the plot belongs to. Only with the matplotlib backend.
    ax : :class:`matplotlib.axes.Axes`
        The axes the plot belongs to. Only with the matplotlib backend.

    Notes
    -----
    This function is only implemented with the matplotlib backend.

    Examples
    --------
    >>> import matplotlib
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> fig, ax = mh.plot()

    """

    if eigenvalues is None:
        eigenvalues = filters.G._e is not None

    if sum is None:
        sum = filters.n_filters > 1

    if labels is None:
        labels = filters.n_filters > 1

    if title is None:
        title = repr(filters)

    return _plt_plot_filter(
        filters,
        n=n,
        eigenvalues=eigenvalues,
        sum=sum,
        labels=labels,
        title=title,
        ax=ax,
        **kwargs,
    )


@_plt_handle_figure
def _plt_plot_filter(filters, n, eigenvalues, sum, labels, ax, **kwargs):
    x = np.linspace(0, filters.G.lmax, n)

    params = dict(alpha=0.5)
    params.update(kwargs)

    if eigenvalues:
        # Evaluate the filter bank at the eigenvalues to avoid plotting
        # artifacts, for example when deltas are centered on the eigenvalues.
        x = np.sort(np.concatenate([x, filters.G.e]))

    y = filters.evaluate(x).T
    lines = ax.plot(x, y, **params)

    # TODO: plot highlighted eigenvalues

    if sum:
        (line_sum,) = ax.plot(x, np.sum(y**2, 1), "k", **kwargs)

    if labels:
        for i, line in enumerate(lines):
            line.set_label(rf"$g_{{{i}}}(\lambda)$")
        if sum:
            line_sum.set_label(r"$\sum_i g_i^2(\lambda)$")
        ax.legend()

    if eigenvalues:
        segs = np.empty((len(filters.G.e), 2, 2))
        segs[:, 0, 0] = segs[:, 1, 0] = filters.G.e
        segs[:, :, 1] = [0, 1]
        mpl, _, _ = _import_plt()
        ax.add_collection(
            mpl.collections.LineCollection(
                segs,
                transform=ax.get_xaxis_transform(),
                zorder=0,
                color=[0.9] * 3,
                linewidth=1,
                label="eigenvalues",
            )
        )

        # Plot dots where the evaluation matters.
        y = filters.evaluate(filters.G.e).T
        params.pop("label", None)
        for i in range(y.shape[1]):
            params.update(color=lines[i].get_color())
            ax.plot(filters.G.e, y[:, i], ".", **params)
        if sum:
            params.update(color=line_sum.get_color())
            ax.plot(filters.G.e, np.sum(y**2, 1), ".", **params)

    ax.set_xlabel(r"laplacian's eigenvalues (graph frequencies) $\lambda$")
    ax.set_ylabel(r"filter response $g(\lambda)$")


def _plot_graph(
    G,
    vertex_color,
    vertex_size,
    highlight,
    edges,
    edge_color,
    edge_width,
    indices,
    colorbar,
    limits,
    ax,
    title,
    backend,
):
    r"""Plot a graph with signals as color or vertex size.

    Parameters
    ----------
    vertex_color : array_like or color
        Signal to plot as vertex color (length is the number of vertices).
        If None, vertex color is set to `graph.plotting['vertex_color']`.
        Alternatively, a color can be set in any format accepted by matplotlib.
        Each vertex color can by specified by an RGB(A) array of dimension
        `n_vertices` x 3 (or 4).
    vertex_size : array_like or int
        Signal to plot as vertex size (length is the number of vertices).
        Vertex size ranges from 0.5 to 2 times `graph.plotting['vertex_size']`.
        If None, vertex size is set to `graph.plotting['vertex_size']`.
        Alternatively, a size can be passed as an integer.
        The pyqtgraph backend only accepts an integer size.
    highlight : iterable
        List of indices of vertices to be highlighted.
        Useful for example to show where a filter was localized.
        Only available with the matplotlib backend.
    edges : bool
        Whether to draw edges in addition to vertices.
        Default to True if less than 10,000 edges to draw.
        Note that drawing many edges can be slow.
    edge_color : array_like or color
        Signal to plot as edge color (length is the number of edges).
        Edge color is given by `graph.plotting['edge_color']` and transparency
        ranges from 0.2 to 0.9.
        If None, edge color is set to `graph.plotting['edge_color']`.
        Alternatively, a color can be set in any format accepted by matplotlib.
        Each edge color can by specified by an RGB(A) array of dimension
        `n_edges` x 3 (or 4).
        Only available with the matplotlib backend.
    edge_width : array_like or int
        Signal to plot as edge width (length is the number of edges).
        Edge width ranges from 0.5 to 2 times `graph.plotting['edge_width']`.
        If None, edge width is set to `graph.plotting['edge_width']`.
        Alternatively, a width can be passed as an integer.
        Only available with the matplotlib backend.
    indices : bool
        Whether to print the node indices (in the adjacency / Laplacian matrix
        and signal vectors) on top of each node.
        Useful to locate a node of interest.
        Only available with the matplotlib backend.
    colorbar : bool
        Whether to plot a colorbar indicating the signal's amplitude.
        Only available with the matplotlib backend.
    limits : [vmin, vmax]
        Map colors from vmin to vmax.
        Defaults to signal minimum and maximum value.
        Only available with the matplotlib backend.
    ax : :class:`matplotlib.axes.Axes`
        Axes where to draw the graph. Optional, created if not passed.
        Only available with the matplotlib backend.
    title : str
        Title of the figure.
    backend: {'matplotlib', 'pyqtgraph', None}
        Defines the drawing backend to use.
        Defaults to :data:`pygsp.plotting.BACKEND`.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The figure the plot belongs to. Only with the matplotlib backend.
    ax : :class:`matplotlib.axes.Axes`
        The axes the plot belongs to. Only with the matplotlib backend.

    Notes
    -----
    The orientation of directed edges is not shown. If edges exist in both
    directions, they will be drawn on top of each other.

    Examples
    --------
    >>> import matplotlib
    >>> graph = graphs.Sensor(20, seed=42)
    >>> graph.compute_fourier_basis(n_eigenvectors=4)
    >>> _, _, weights = graph.get_edge_list()
    >>> fig, ax = graph.plot(graph.U[:, 1], vertex_size=graph.dw,
    ...                      edge_color=weights)
    >>> graph.plotting['vertex_size'] = 300
    >>> graph.plotting['edge_width'] = 5
    >>> graph.plotting['edge_style'] = '--'
    >>> fig, ax = graph.plot(edge_width=weights, edge_color=(0, .8, .8, .5),
    ...                      vertex_color='black')
    >>> fig, ax = graph.plot(vertex_size=graph.dw, indices=True,
    ...                      highlight=[17, 3, 16], edges=False)

    """
    if not hasattr(G, "coords") or G.coords is None:
        raise AttributeError(
            "Graph has no coordinate set. " "Please run G.set_coordinates() first."
        )
    check_2d_3d = (G.coords.ndim != 2) or (G.coords.shape[1] not in [2, 3])
    if G.coords.ndim != 1 and check_2d_3d:
        raise AttributeError("Coordinates should be in 1D, 2D or 3D space.")
    if G.coords.shape[0] != G.N:
        raise AttributeError(f"Graph needs G.N = {G.N} coordinates.")

    if backend is None:
        backend = BACKEND

    def check_shape(signal, name, length, many=False):
        if (signal.ndim == 0) or (signal.shape[0] != length):
            txt = "{}: signal should have length {}."
            txt = txt.format(name, length)
            raise ValueError(txt)
        if (not many) and (signal.ndim != 1):
            txt = "{}: can plot only one signal (not {})."
            txt = txt.format(name, signal.shape[1])
            raise ValueError(txt)

    def normalize(x):
        """Scale values in [intercept, 1]. Return 0.5 if constant.

        Set intercept value in G.plotting["normalize_intercept"]
        with value in [0, 1], default is .25.
        """
        ptp = np.ptp(x)
        if ptp == 0:
            return np.full(x.shape, 0.5)
        else:
            intercept = G.plotting["normalize_intercept"]
            return (1.0 - intercept) * (x - x.min()) / ptp + intercept

    def is_color(color):
        if backend == "matplotlib":
            mpl, _, _ = _import_plt()
            if mpl.colors.is_color_like(color):
                return True  # single color
            try:
                return all(map(mpl.colors.is_color_like, color))  # color list
            except TypeError:
                return False  # e.g., color is an int

        else:
            return False  # No support for pyqtgraph (yet).

    if vertex_color is None:
        limits = [0, 0]
        colorbar = False
        if backend == "matplotlib":
            vertex_color = (G.plotting["vertex_color"],)
    elif is_color(vertex_color):
        limits = [0, 0]
        colorbar = False
    else:
        vertex_color = np.asanyarray(vertex_color).squeeze()
        check_shape(
            vertex_color, "Vertex color", G.n_vertices, many=(G.coords.ndim == 1)
        )

    if vertex_size is None:
        vertex_size = G.plotting["vertex_size"]
    elif not np.isscalar(vertex_size):
        vertex_size = np.asanyarray(vertex_size).squeeze()
        check_shape(vertex_size, "Vertex size", G.n_vertices)
        vertex_size = G.plotting["vertex_size"] * 4 * normalize(vertex_size) ** 2

    if edges is None:
        edges = G.Ne < 10e3

    if edge_color is None:
        edge_color = (G.plotting["edge_color"],)
    elif not is_color(edge_color):
        edge_color = np.asanyarray(edge_color).squeeze()
        check_shape(edge_color, "Edge color", G.n_edges)
        edge_color = 0.9 * normalize(edge_color)
        edge_color = [
            np.tile(G.plotting["edge_color"][:3], [len(edge_color), 1]),
            edge_color[:, np.newaxis],
        ]
        edge_color = np.concatenate(edge_color, axis=1)

    if edge_width is None:
        edge_width = G.plotting["edge_width"]
    elif not np.isscalar(edge_width):
        edge_width = np.array(edge_width).squeeze()
        check_shape(edge_width, "Edge width", G.n_edges)
        edge_width = G.plotting["edge_width"] * 2 * normalize(edge_width)

    if limits is None:
        limits = [1.05 * vertex_color.min(), 1.05 * vertex_color.max()]

    if title is None:
        title = G.__repr__(limit=4)

    if backend == "pyqtgraph":
        if vertex_color is None:
            _qtg_plot_graph(G, edges=edges, vertex_size=vertex_size, title=title)
        else:
            _qtg_plot_signal(
                G,
                signal=vertex_color,
                vertex_size=vertex_size,
                edges=edges,
                limits=limits,
                title=title,
            )
    elif backend == "matplotlib":
        return _plt_plot_graph(
            G,
            vertex_color=vertex_color,
            vertex_size=vertex_size,
            highlight=highlight,
            edges=edges,
            indices=indices,
            colorbar=colorbar,
            edge_color=edge_color,
            edge_width=edge_width,
            limits=limits,
            ax=ax,
            title=title,
        )
    else:
        raise ValueError(f"Unknown backend {backend}.")


@_plt_handle_figure
def _plt_plot_graph(
    G,
    vertex_color,
    vertex_size,
    highlight,
    edges,
    edge_color,
    edge_width,
    indices,
    colorbar,
    limits,
    ax,
):
    mpl, plt, mplot3d = _import_plt()

    if edges and (G.coords.ndim != 1):  # No edges for 1D plots.
        sources, targets, _ = G.get_edge_list()
        edges = [
            G.coords[sources],
            G.coords[targets],
        ]
        edges = np.stack(edges, axis=1)

        if G.coords.shape[1] == 2:
            LineCollection = mpl.collections.LineCollection
        elif G.coords.shape[1] == 3:
            LineCollection = mplot3d.art3d.Line3DCollection
        ax.add_collection(
            LineCollection(
                edges,
                linewidths=edge_width,
                colors=edge_color,
                linestyles=G.plotting["edge_style"],
                zorder=1,
            )
        )

    try:
        iter(highlight)
    except TypeError:
        highlight = [highlight]
    coords_hl = G.coords[highlight]

    if G.coords.ndim == 1:
        ax.plot(G.coords, vertex_color, alpha=0.5)
        ax.set_ylim(limits)
        for coord_hl in coords_hl:
            ax.axvline(x=coord_hl, color=G.plotting["highlight_color"], linewidth=2)

    else:
        sc = ax.scatter(
            *G.coords.T,
            c=vertex_color,
            s=vertex_size,
            marker="o",
            linewidths=0,
            alpha=0.5,
            zorder=2,
            vmin=limits[0],
            vmax=limits[1],
        )
        if np.isscalar(vertex_size):
            size_hl = vertex_size
        else:
            size_hl = vertex_size[highlight]
        ax.scatter(
            *coords_hl.T,
            s=2 * size_hl,
            zorder=3,
            marker="o",
            c="None",
            edgecolors=G.plotting["highlight_color"],
            linewidths=2,
        )

        if G.coords.shape[1] == 3:
            try:
                ax.view_init(elev=G.plotting["elevation"], azim=G.plotting["azimuth"])
                ax.dist = G.plotting["distance"]
            except KeyError:
                pass

    if G.coords.ndim != 1 and colorbar:
        plt.colorbar(sc, ax=ax)

    if indices:
        for node in range(G.N):
            ax.text(
                *tuple(G.coords[node]),  # accomodate 2D and 3D
                s=node,
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )


def _qtg_plot_signal(G, signal, edges, vertex_size, limits, title):
    qtg, gl, QtGui, QtWidgets = _import_qtg()

    if G.coords.shape[1] == 2:
        widget = qtg.GraphicsLayoutWidget()
        view = widget.addViewBox()

    elif G.coords.shape[1] == 3:
        QApplication = _get_qapplication(QtGui, QtWidgets)
        if not QApplication.instance():
            QApplication([])  # We want only one application.
        widget = gl.GLViewWidget()
        widget.opts["distance"] = 10

    if edges:
        if G.coords.shape[1] == 2:
            adj = _get_coords(G, edge_list=True)
            pen = tuple(np.array(G.plotting["edge_color"]) * 255)
            g = qtg.GraphItem(
                pos=G.coords, adj=adj, symbolBrush=None, symbolPen=None, pen=pen
            )
            view.addItem(g)

        elif G.coords.shape[1] == 3:
            x, y, z = _get_coords(G)
            pos = np.stack((x, y, z), axis=1)
            g = gl.GLLinePlotItem(pos=pos, mode="lines", color=G.plotting["edge_color"])
            widget.addItem(g)

    pos = [1, 8, 24, 40, 56, 64]
    color = np.array(
        [
            [0, 0, 143, 255],
            [0, 0, 255, 255],
            [0, 255, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 0, 255],
            [128, 0, 0, 255],
        ]
    )
    cmap = qtg.ColorMap(pos, color)

    signal = 1 + 63 * (signal - limits[0]) / limits[1] - limits[0]

    if G.coords.shape[1] == 2:
        gp = qtg.ScatterPlotItem(
            G.coords[:, 0],
            G.coords[:, 1],
            size=vertex_size / 10,
            brush=cmap.map(signal, "qcolor"),
        )
        view.addItem(gp)

    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(
            pos=G.coords, size=vertex_size / 3, color=cmap.map(signal, "float")
        )
        widget.addItem(gp)

    widget.setWindowTitle(title)
    widget.show()

    global _qtg_widgets
    _qtg_widgets.append(widget)


def _plot_spectrogram(G, node_idx):
    r"""Plot the graph's spectrogram.

    Parameters
    ----------
    node_idx : ndarray
        Order to sort the nodes in the spectrogram.
        By default, does not reorder the nodes.

    Notes
    -----
    This function is only implemented for the pyqtgraph backend at the moment.

    Examples
    --------
    >>> G = graphs.Ring(15)
    >>> G.plot_spectrogram()

    """
    from pygsp import features

    qtg, _, _ = _import_qtg()

    if not hasattr(G, "spectr"):
        features.compute_spectrogram(G)

    M = G.spectr.shape[1]
    spectr = G.spectr[node_idx, :] if node_idx is not None else G.spectr
    spectr = np.ravel(spectr)
    min_spec, max_spec = spectr.min(), spectr.max()

    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    color = [
        [20, 133, 212, 255],
        [53, 42, 135, 255],
        [48, 174, 170, 255],
        [210, 184, 87, 255],
        [249, 251, 14, 255],
    ]
    color = np.array(color, dtype=np.ubyte)
    cmap = qtg.ColorMap(pos, color)

    spectr = (spectr.astype(float) - min_spec) / (max_spec - min_spec)

    widget = qtg.GraphicsLayoutWidget()
    label = f"frequencies {0}:{G.lmax / M:.2f}:{G.lmax:.2f}"
    v = widget.addPlot(labels={"bottom": "nodes", "left": label})
    v.setAspectLocked()

    spi = qtg.ScatterPlotItem(
        np.repeat(np.arange(G.N), M),
        np.ravel(np.tile(np.arange(M), (1, G.N))),
        pxMode=False,
        symbol="s",
        size=1,
        brush=cmap.map(spectr, "qcolor"),
    )
    v.addItem(spi)

    widget.setWindowTitle(f"Spectrogram of {G.__repr__(limit=4)}")
    widget.show()

    global _qtg_widgets
    _qtg_widgets.append(widget)


def _get_coords(G, edge_list=False):
    sources, targets, _ = G.get_edge_list()

    if edge_list:
        return np.stack((sources, targets), axis=1)

    coords = [
        np.stack((G.coords[sources, d], G.coords[targets, d]), axis=0)
        for d in range(G.coords.shape[1])
    ]

    if G.coords.shape[1] == 2:
        return coords

    elif G.coords.shape[1] == 3:
        return [coord.reshape(-1, order="F") for coord in coords]
