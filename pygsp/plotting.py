# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.plotting` module implements functionality to plot PyGSP objects
with a `pyqtgraph <http://www.pyqtgraph.org>`_ or `matplotlib
<https://matplotlib.org>`_ drawing backend (which can be controlled by the
:data:`BACKEND` constant or individually for each plotting call).

Most users won't use this module directly.
Graphs (from :mod:`pygsp.graphs`) are to be plotted with
:meth:`pygsp.graphs.Graph.plot`, :meth:`pygsp.graphs.Graph.plot_spectrogram`,
and :meth:`pygsp.graphs.Graph.plot_signal`.
Filters (from :mod:`pygsp.filters`) are to be plotted with
:meth:`pygsp.filters.Filter.plot`.

.. data:: BACKEND

    Indicates which drawing backend to use if none are provided to the plotting
    functions. Should be either ``'matplotlib'`` or ``'pyqtgraph'``. In general
    pyqtgraph is better for interactive exploration while matplotlib is better
    at generating figures to be included in papers or elsewhere.

"""

from __future__ import division

import functools

import numpy as np

from pygsp import utils


_logger = utils.build_logger(__name__)

BACKEND = 'matplotlib'
_qtg_windows = []
_qtg_widgets = []
_plt_figures = []


def _import_plt():
    try:
        import matplotlib.pyplot as plt
        # Not used directly, but needed for 3D projection.
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except Exception:
        raise ImportError('Cannot import matplotlib. Choose another backend '
                          'or try to install it with '
                          'pip (or conda) install matplotlib.')
    return plt


def _import_qtg():
    try:
        import pyqtgraph as qtg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtGui
    except Exception:
        raise ImportError('Cannot import pyqtgraph. Choose another backend '
                          'or try to install it with '
                          'pip (or conda) install pyqtgraph. You will also '
                          'need PyQt5 (or PySide) and PyOpenGL.')
    return qtg, gl, QtGui


def _plt_handle_figure(plot):
    r"""Handle the common work (creating an axis if not given, setting the
    title, saving the created plot) of all matplotlib plot commands."""

    # Preserve documentation of plot.
    @functools.wraps(plot)

    def inner(obj, **kwargs):

        plt = _import_plt()

        # Create a figure and an axis if none were passed.
        if kwargs['ax'] is None:
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)

            if (hasattr(obj, 'coords') and obj.coords.ndim == 2 and
                    obj.coords.shape[1] == 3):
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            kwargs['ax'] = ax

        save = kwargs.pop('save')
        title = kwargs.pop('title')

        plot(obj, **kwargs)

        kwargs['ax'].set_title(title)

        try:
            if save is not None:
                fig.savefig(save + '.png')
                fig.savefig(save + '.pdf')
            else:
                fig.show(warn=False)
        except NameError:
            # No figure created, an axis was passed.
            pass

    return inner


def close_all():
    r"""Close all opened windows."""

    # Windows can be closed by releasing all references to them so they can be
    # garbage collected. May not be necessary to call close().
    global _qtg_windows
    for window in _qtg_windows:
        window.close()
    _qtg_windows = []

    global _qtg_widgets
    for widget in _qtg_widgets:
        widget.close()
    _qtg_widgets = []

    global _plt_figures
    for fig in _plt_figures:
        plt = _import_plt()
        plt.close(fig)
    _plt_figures = []


def show(*args, **kwargs):
    r"""Show created figures, alias to plt.show().

    By default, showing plots does not block the prompt.
    """
    plt = _import_plt()
    plt.show(*args, **kwargs)


def close(*args, **kwargs):
    r"""Close created figures, alias to plt.close()."""
    plt = _import_plt()
    plt.close(*args, **kwargs)


def _plot_graph(G, edges, backend, vertex_size, title, save, ax):
    r"""Plot the graph.

    Parameters
    ----------
    edges : bool
        True to draw edges, false to only draw vertices.
        Default True if less than 10,000 edges to draw.
        Note that drawing a large number of edges might be particularly slow.
    backend: {'matplotlib', 'pyqtgraph'}
        Defines the drawing backend to use. Defaults to :data:`BACKEND`.
    vertex_size : float
        Size of circle representing each node.
        Defaults to G.plotting['vertex_size'].
    title : str
        Title of the figure.
    save : str
        Whether to save the plot as save.png and save.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed.
        Only available with the matplotlib backend.

    Examples
    --------
    >>> import matplotlib
    >>> G = graphs.Logo()
    >>> G.plot()

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')
    if (G.coords.ndim != 2) or (G.coords.shape[1] not in [2, 3]):
        raise AttributeError('Coordinates should be in 2D or 3D space.')

    if edges is None:
        edges = G.Ne < 10e3

    if backend is None:
        backend = BACKEND

    if vertex_size is None:
        vertex_size = G.plotting['vertex_size']

    if title is None:
        title = u'{}\nG.N={} nodes, G.Ne={} edges'.format(G.gtype, G.N, G.Ne)

    G = _handle_directed(G)

    if backend == 'pyqtgraph':
        _qtg_plot_graph(G, edges=edges, vertex_size=vertex_size, title=title)
    elif backend == 'matplotlib':
        _plt_plot_graph(G, edges=edges, vertex_size=vertex_size, title=title,
                        save=save, ax=ax)
    else:
        raise ValueError('Unknown backend {}.'.format(backend))


@_plt_handle_figure
def _plt_plot_graph(G, edges, vertex_size, ax):

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    if edges:

        if G.is_directed():
            raise NotImplementedError

        else:

            if G.coords.shape[1] == 2:
                x, y = _get_coords(G)
                ax.plot(x, y, linewidth=G.plotting['edge_width'],
                        color=G.plotting['edge_color'],
                        linestyle=G.plotting['edge_style'],
                        marker='o', markersize=vertex_size/10,
                        markerfacecolor=G.plotting['vertex_color'],
                        markeredgecolor=G.plotting['vertex_color'])

            if G.coords.shape[1] == 3:
                # TODO: very dirty. Cannot we prepare a set of lines?
                x, y, z = _get_coords(G)
                for i in range(0, x.size, 2):
                    x2, y2, z2 = x[i:i+2], y[i:i+2], z[i:i+2]
                    ax.plot(x2, y2, z2, linewidth=G.plotting['edge_width'],
                            color=G.plotting['edge_color'],
                            linestyle=G.plotting['edge_style'],
                            marker='o', markersize=vertex_size/10,
                            markerfacecolor=G.plotting['vertex_color'],
                            markeredgecolor=G.plotting['vertex_color'])

    else:

        # TODO: is ax.plot(G.coords[:, 0], G.coords[:, 1], 'bo') faster?
        if G.coords.shape[1] == 2:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], marker='o',
                       s=vertex_size,
                       c=G.plotting['vertex_color'])

        if G.coords.shape[1] == 3:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2],
                       marker='o', s=vertex_size,
                       c=G.plotting['vertex_color'])

    if G.coords.shape[1] == 3:
        try:
            ax.view_init(elev=G.plotting['elevation'],
                         azim=G.plotting['azimuth'])
            ax.dist = G.plotting['distance']
        except KeyError:
            pass


def _qtg_plot_graph(G, edges, vertex_size, title):

    # TODO handling when G is a list of graphs

    qtg, gl, QtGui = _import_qtg()

    if G.is_directed():
        raise NotImplementedError

    else:

        if G.coords.shape[1] == 2:

            window = qtg.GraphicsWindow()
            window.setWindowTitle(title)
            view = window.addViewBox()
            view.setAspectLocked()

            if edges:
                pen = tuple(np.array(G.plotting['edge_color']) * 255)
            else:
                pen = None

            adj = _get_coords(G, edge_list=True)

            g = qtg.GraphItem(pos=G.coords, adj=adj, pen=pen,
                              size=vertex_size/10)
            view.addItem(g)

            global _qtg_windows
            _qtg_windows.append(window)

        elif G.coords.shape[1] == 3:
            if not QtGui.QApplication.instance():
                QtGui.QApplication([])  # We want only one application.
            widget = gl.GLViewWidget()
            widget.opts['distance'] = 10
            widget.show()
            widget.setWindowTitle(title)

            if edges:
                x, y, z = _get_coords(G)
                pos = np.stack((x, y, z), axis=1)
                g = gl.GLLinePlotItem(pos=pos, mode='lines',
                                      color=G.plotting['edge_color'])
                widget.addItem(g)

            gp = gl.GLScatterPlotItem(pos=G.coords, size=vertex_size/3,
                                      color=G.plotting['vertex_color'])
            widget.addItem(gp)

            global _qtg_widgets
            _qtg_widgets.append(widget)


def _plot_filter(filters, n, eigenvalues, sum, title, save, ax, **kwargs):
    r"""Plot the spectral response of a filter bank, a set of graph filters.

    Parameters
    ----------
    n : int
        Number of points where the filters are evaluated.
    eigenvalues : boolean
        To plot black X marks at all eigenvalues of the graph. You need to
        compute the Fourier basis to use this option. By default the
        eigenvalues are plot if they are contained in the Graph.
    sum : boolean
        To plot an extra line showing the sum of the squared magnitudes
        of the filters (default True if there is multiple filters).
    title : str
        Title of the figure.
    save : str
        Whether to save the plot as save.png and save.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed.
        Only available with the matplotlib backend.
    kwargs : dict
        Additional parameters passed to the matplotlib plot function.
        Useful for example to change the linewidth, linestyle, or set a label.
        Only available with the matplotlib backend.

    Notes
    -----
    This function is only implemented for the matplotlib backend at the moment.

    Examples
    --------
    >>> import matplotlib
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> mh.plot()

    """

    if eigenvalues is None:
        eigenvalues = hasattr(filters.G, '_e')

    if sum is None:
        sum = filters.n_filters > 1

    if title is None:
        title = repr(filters)

    _plt_plot_filter(filters, n=n, eigenvalues=eigenvalues, sum=sum,
                     title=title, save=save, ax=ax, **kwargs)


@_plt_handle_figure
def _plt_plot_filter(filters, n, eigenvalues, sum, ax, **kwargs):

    if eigenvalues:
        for e in filters.G.e:
            ax.axvline(x=e, color=[0.9]*3, linewidth=1)

    x = np.linspace(0, filters.G.lmax, n)
    y = filters.evaluate(x).T
    ax.plot(x, y, **kwargs)

    # TODO: plot highlighted eigenvalues

    if sum:
        ax.plot(x, np.sum(y**2, 1), 'k', **kwargs)

    ax.set_xlabel(r"$\lambda$: laplacian's eigenvalues / graph frequencies")
    ax.set_ylabel(r'$\hat{g}(\lambda)$: filter response')


def _plot_signal(G, signal, edges, vertex_size, highlight, colorbar,
                 limits, backend, title, save, ax):
    r"""Plot a signal on the graph.

    Parameters
    ----------
    signal : array of int
        Signal to plot. Signal length should be equal to the number of nodes.
    edges : bool
        True to draw edges, false to only draw vertices.
        Default True if less than 10,000 edges to draw.
        Note that drawing a large number of edges might be particularly slow.
    cp : list of int
        NOT IMPLEMENTED. Camera position when plotting a 3D graph.
    vertex_size : float
        Size of circle representing each node.
        Defaults to G.plotting['vertex_size'].
    highlight : iterable
        List of indices of vertices to be highlighted.
        Useful to e.g. show where a filter was localized.
        Only available with the matplotlib backend.
    colorbar : bool
        Whether to plot a colorbar indicating the signal's amplitude.
        Only available with the matplotlib backend.
    limits : [vmin, vmax]
        Maps colors from vmin to vmax.
        Defaults to signal minimum and maximum value.
        Only available with the matplotlib backend.
    bar : boolean
        NOT IMPLEMENTED. Signal values are displayed using colors when False,
        and bars when True (default False).
    bar_width : int
        NOT IMPLEMENTED. Width of the bar (default 1).
    backend: {'matplotlib', 'pyqtgraph'}
        Defines the drawing backend to use. Defaults to :data:`BACKEND`.
    title : str
        Title of the figure.
    save : str
        Whether to save the plot as save.png and save.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed.
        Only available with the matplotlib backend.

    Examples
    --------
    >>> import matplotlib
    >>> G = graphs.Grid2d(8)
    >>> signal = np.sin((np.arange(8**2) * 2*np.pi/8**2))
    >>> G.plot_signal(signal)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')
    check_2d_3d = (G.coords.ndim != 2) or (G.coords.shape[1] not in [2, 3])
    if G.coords.ndim != 1 and check_2d_3d:
        raise AttributeError('Coordinates should be in 1D, 2D or 3D space.')

    signal = signal.squeeze()
    if G.coords.ndim == 2 and signal.ndim != 1:
        raise ValueError('Can plot only one signal (not {}) with {}D '
                         'coordinates.'.format(signal.shape[1],
                                               G.coords.shape[1]))
    if signal.shape[0] != G.N:
        raise ValueError('Signal length is {}, should be '
                         'G.N = {}.'.format(signal.shape[0], G.N))
    if np.sum(np.abs(signal.imag)) > 1e-10:
        raise ValueError("Can't display complex signals.")

    if edges is None:
        edges = G.Ne < 10e3

    if vertex_size is None:
        vertex_size = G.plotting['vertex_size']

    if title is None:
        title = u'{}\nG.N={} nodes, G.Ne={} edges'.format(G.gtype, G.N, G.Ne)

    if limits is None:
        limits = [1.05*signal.min(), 1.05*signal.max()]

    if backend is None:
        backend = BACKEND

    G = _handle_directed(G)

    if backend == 'pyqtgraph':
        _qtg_plot_signal(G, signal=signal, edges=edges,
                         vertex_size=vertex_size, limits=limits, title=title)
    elif backend == 'matplotlib':
        _plt_plot_signal(G, signal=signal, edges=edges,
                         vertex_size=vertex_size, limits=limits, title=title,
                         highlight=highlight, colorbar=colorbar,
                         save=save, ax=ax)
    else:
        raise ValueError('Unknown backend {}.'.format(backend))


@_plt_handle_figure
def _plt_plot_signal(G, signal, edges, vertex_size, highlight, colorbar,
                     limits, ax):

    if edges:

        if G.is_directed():
            raise NotImplementedError

        else:

            if G.coords.ndim == 1:
                pass

            elif G.coords.shape[1] == 2:
                x, y = _get_coords(G)
                ax.plot(x, y, linewidth=G.plotting['edge_width'],
                        color=G.plotting['edge_color'],
                        linestyle=G.plotting['edge_style'],
                        zorder=1)

            elif G.coords.shape[1] == 3:
                # TODO: very dirty. Cannot we prepare a set of lines?
                x, y, z = _get_coords(G)
                for i in range(0, x.size, 2):
                    x2, y2, z2 = x[i:i+2], y[i:i+2], z[i:i+2]
                    ax.plot(x2, y2, z2, linewidth=G.plotting['edge_width'],
                            color=G.plotting['edge_color'],
                            linestyle=G.plotting['edge_style'],
                            zorder=1)

    try:
        iter(highlight)
    except TypeError:
        highlight = [highlight]
    coords_hl = G.coords[highlight]

    if G.coords.ndim == 1:
        ax.plot(G.coords, signal)
        ax.set_ylim(limits)
        for coord_hl in coords_hl:
            ax.axvline(x=coord_hl, color='C1', linewidth=2)

    elif G.coords.shape[1] == 2:
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1],
                        s=vertex_size, c=signal, zorder=2,
                        vmin=limits[0], vmax=limits[1])
        ax.scatter(coords_hl[:, 0], coords_hl[:, 1],
                   s=2*vertex_size, zorder=3,
                   marker='o', c='None', edgecolors='C1', linewidths=2)

    elif G.coords.shape[1] == 3:
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2],
                        s=vertex_size, c=signal, zorder=2,
                        vmin=limits[0], vmax=limits[1])
        ax.scatter(coords_hl[:, 0], coords_hl[:, 1], coords_hl[:, 2],
                   s=2*vertex_size, zorder=3,
                   marker='o', c='None', edgecolors='C1', linewidths=2)
        try:
            ax.view_init(elev=G.plotting['elevation'],
                         azim=G.plotting['azimuth'])
            ax.dist = G.plotting['distance']
        except KeyError:
            pass

    if G.coords.ndim != 1 and colorbar:
        plt = _import_plt()
        plt.colorbar(sc, ax=ax)


def _qtg_plot_signal(G, signal, edges, vertex_size, limits, title):

    qtg, gl, QtGui = _import_qtg()

    if G.coords.shape[1] == 2:
        window = qtg.GraphicsWindow(title)
        view = window.addViewBox()

    elif G.coords.shape[1] == 3:
        if not QtGui.QApplication.instance():
            QtGui.QApplication([])  # We want only one application.
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle(title)

    if edges:

        if G.is_directed():
            raise NotImplementedError

        else:

            if G.coords.shape[1] == 2:
                adj = _get_coords(G, edge_list=True)
                pen = tuple(np.array(G.plotting['edge_color']) * 255)
                g = qtg.GraphItem(pos=G.coords, adj=adj, symbolBrush=None,
                                  symbolPen=None, pen=pen)
                view.addItem(g)

            elif G.coords.shape[1] == 3:
                x, y, z = _get_coords(G)
                pos = np.stack((x, y, z), axis=1)
                g = gl.GLLinePlotItem(pos=pos, mode='lines',
                                      color=G.plotting['edge_color'])
                widget.addItem(g)

    pos = [1, 8, 24, 40, 56, 64]
    color = np.array([[0, 0, 143, 255], [0, 0, 255, 255], [0, 255, 255, 255],
                      [255, 255, 0, 255], [255, 0, 0, 255], [128, 0, 0, 255]])
    cmap = qtg.ColorMap(pos, color)

    signal = 1 + 63 * (signal - limits[0]) / limits[1] - limits[0]

    if G.coords.shape[1] == 2:
        gp = qtg.ScatterPlotItem(G.coords[:, 0],
                                 G.coords[:, 1],
                                 size=vertex_size/10,
                                 brush=cmap.map(signal, 'qcolor'))
        view.addItem(gp)

    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(pos=G.coords,
                                  size=vertex_size/3,
                                  color=cmap.map(signal, 'float'))
        widget.addItem(gp)

    if G.coords.shape[1] == 2:
        global _qtg_windows
        _qtg_windows.append(window)
    elif G.coords.shape[1] == 3:
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

    if not hasattr(G, 'spectr'):
        features.compute_spectrogram(G)

    M = G.spectr.shape[1]
    spectr = G.spectr[node_idx, :] if node_idx is not None else G.spectr
    spectr = np.ravel(spectr)
    min_spec, max_spec = spectr.min(), spectr.max()

    pos = np.array([0., 0.25, 0.5, 0.75, 1.])
    color = [[20, 133, 212, 255], [53, 42, 135, 255], [48, 174, 170, 255],
             [210, 184, 87, 255], [249, 251, 14, 255]]
    color = np.array(color, dtype=np.ubyte)
    cmap = qtg.ColorMap(pos, color)

    spectr = (spectr.astype(float) - min_spec) / (max_spec - min_spec)

    w = qtg.GraphicsWindow()
    w.setWindowTitle("Spectrogram of {}".format(G.gtype))
    label = 'frequencies {}:{:.2f}:{:.2f}'.format(0, G.lmax/M, G.lmax)
    v = w.addPlot(labels={'bottom': 'nodes',
                          'left': label})
    v.setAspectLocked()

    spi = qtg.ScatterPlotItem(np.repeat(np.arange(G.N), M),
                              np.ravel(np.tile(np.arange(M), (1, G.N))),
                              pxMode=False,
                              symbol='s',
                              size=1,
                              brush=cmap.map(spectr, 'qcolor'))
    v.addItem(spi)

    global _qtg_windows
    _qtg_windows.append(w)


def _get_coords(G, edge_list=False):

    v_in, v_out, _ = G.get_edge_list()

    if edge_list:
        return np.stack((v_in, v_out), axis=1)

    coords = [np.stack((G.coords[v_in, d], G.coords[v_out, d]), axis=0)
              for d in range(G.coords.shape[1])]

    if G.coords.shape[1] == 2:
        return coords

    elif G.coords.shape[1] == 3:
        return [coord.reshape(-1, order='F') for coord in coords]


def _handle_directed(G):
    # FIXME: plot edge direction. For now we just symmetrize the weight matrix.
    if not G.is_directed():
        return G
    else:
        from pygsp import graphs
        G2 = graphs.Graph(utils.symmetrize(G.W))
        G2.coords = G.coords
        G2.plotting = G.plotting
        return G2
