# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.plotting` module implements functionality to plot PyGSP objects
with a `pyqtgraph <http://www.pyqtgraph.org>`_ or `matplotlib
<https://matplotlib.org>`_ drawing backend (which can be controlled by the
:data:`BACKEND` constant or individually for each plotting call):

* graphs from :mod:`pygsp.graphs` with :func:`plot_graph`,
  :func:`plot_spectrogram`, and :func:`plot_signal`,
* filters from :mod:`pygsp.filters` with :func:`plot_filter`.

.. data:: BACKEND

    Indicates which drawing backend to use if none are provided to the plotting
    functions. Should be either 'matplotlib' or 'pyqtgraph'. In general
    pyqtgraph is better for interactive exploration while matplotlib is better
    at generating figures to be included in papers or elsewhere.

"""

from __future__ import division

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

    def inner(obj, *args, **kwargs):

        plt = _import_plt()

        # Create a figure and an axis if none were passed.
        if 'ax' not in kwargs.keys():
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)

            if (hasattr(obj, 'coords') and obj.coords.ndim == 2 and
                    obj.coords.shape[1] == 3):
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            kwargs.update(ax=ax)

        save_as = kwargs.pop('save_as', None)
        plot_name = kwargs.pop('plot_name', '')

        plot(obj, *args, **kwargs)

        kwargs['ax'].set_title(plot_name)

        try:
            if save_as is not None:
                fig.savefig(save_as + '.png')
                fig.savefig(save_as + '.pdf')
            else:
                fig.show(warn=False)
        except NameError:
            # No figure created, an axis was passed.
            pass

    return inner


def close_all():
    r"""
    Close all opened windows.

    """

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
    r"""
    Show created figures.

    Alias to plt.show().
    By default, showing plots does not block the prompt.

    """
    plt = _import_plt()
    plt.show(*args, **kwargs)


def close(*args, **kwargs):
    r"""
    Close created figures.

    Alias to plt.close().

    """
    plt = _import_plt()
    plt.close(*args, **kwargs)


def plot(O, **kwargs):
    r"""
    Main plotting function.

    This convenience function either calls :func:`plot_graph` or
    :func:`plot_filter` given the type of the passed object. Parameters can be
    passed to those functions.

    Parameters
    ----------
    O : Graph, Filter
        object to plot

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Logo()
    >>> plotting.plot(G)

    """

    try:
        O.plot(**kwargs)
    except AttributeError:
        raise TypeError('Unrecognized object, i.e. not a Graph or Filter.')


def plot_graph(G, backend=None, **kwargs):
    r"""
    Plot a graph or a list of graphs.

    Parameters
    ----------
    G : Graph
        Graph to plot.
    show_edges : bool
        True to draw edges, false to only draw vertices.
        Default True if less than 10,000 edges to draw.
        Note that drawing a large number of edges might be particularly slow.
    backend: {'matplotlib', 'pyqtgraph'}
        Defines the drawing backend to use. Defaults to :data:`BACKEND`.
    vertex_size : float
        Size of circle representing each node.
    plot_name : str
        name of the plot
    save_as : str
        Whether to save the plot as save_as.png and save_as.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed. Only
        available with the matplotlib backend.

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Logo()
    >>> plotting.plot_graph(G)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')
    if (G.coords.ndim != 2) or (G.coords.shape[1] not in [2, 3]):
        raise AttributeError('Coordinates should be in 2D or 3D space.')

    kwargs['show_edges'] = kwargs.pop('show_edges', G.Ne < 10e3)

    default = G.plotting['vertex_size']
    kwargs['vertex_size'] = kwargs.pop('vertex_size', default)

    plot_name = u'{}\nG.N={} nodes, G.Ne={} edges'.format(G.gtype, G.N, G.Ne)
    kwargs['plot_name'] = kwargs.pop('plot_name', plot_name)

    if backend is None:
        backend = BACKEND

    G = _handle_directed(G)

    if backend == 'pyqtgraph':
        _qtg_plot_graph(G, **kwargs)
    elif backend == 'matplotlib':
        _plt_plot_graph(G, **kwargs)
    else:
        raise ValueError('Unknown backend {}.'.format(backend))


@_plt_handle_figure
def _plt_plot_graph(G, show_edges, vertex_size, ax):

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    if show_edges:

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


def _qtg_plot_graph(G, show_edges, vertex_size, plot_name):

    # TODO handling when G is a list of graphs

    qtg, gl, QtGui = _import_qtg()

    if G.is_directed():
        raise NotImplementedError

    else:

        if G.coords.shape[1] == 2:

            window = qtg.GraphicsWindow()
            window.setWindowTitle(plot_name)
            view = window.addViewBox()
            view.setAspectLocked()

            if show_edges:
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
            widget.setWindowTitle(plot_name)

            if show_edges:
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


@_plt_handle_figure
def plot_filter(filters, npoints=1000, line_width=4, x_width=3,
                x_size=10, plot_eigenvalues=None, show_sum=None, ax=None):
    r"""
    Plot the spectral response of a filter bank, a set of graph filters.

    Parameters
    ----------
    filters : Filter
        Filter bank to plot.
    npoints : int
        Number of point where the filters are evaluated.
    line_width : int
        Width of the filters plots.
    x_width : int
        Width of the X marks representing the eigenvalues.
    x_size : int
        Size of the X marks representing the eigenvalues.
    plot_eigenvalues : boolean
        To plot black X marks at all eigenvalues of the graph. You need to
        compute the Fourier basis to use this option. By default the
        eigenvalues are plot if they are contained in the Graph.
    show_sum : boolean
        To plot an extra line showing the sum of the squared magnitudes
        of the filters (default True if there is multiple filters).
    plot_name : string
        name of the plot
    save_as : str
        Whether to save the plot as save_as.png and save_as.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed. Only
        available with the matplotlib backend.

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> plotting.plot_filter(mh)

    """

    G = filters.G

    if plot_eigenvalues is None:
        plot_eigenvalues = hasattr(G, '_e')
    if show_sum is None:
        show_sum = filters.Nf > 1

    if plot_eigenvalues:
        for e in G.e:
            ax.axvline(x=e, color=[0.9]*3, linewidth=1)

    x = np.linspace(0, G.lmax, npoints)
    y = filters.evaluate(x).T
    ax.plot(x, y, linewidth=line_width)

    # TODO: plot highlighted eigenvalues

    if show_sum:
        ax.plot(x, np.sum(y**2, 1), 'k', linewidth=line_width)

    ax.set_xlabel("$\lambda$: laplacian's eigenvalues / graph frequencies")
    ax.set_ylabel('$\hat{g}(\lambda)$: filter response')


def plot_signal(G, signal, backend=None, **kwargs):
    r"""
    Plot a signal on top of a graph.

    Parameters
    ----------
    G : Graph
        Graph to plot a signal on top.
    signal : array of int
        Signal to plot. Signal length should be equal to the number of nodes.
    show_edges : bool
        True to draw edges, false to only draw vertices.
        Default True if less than 10,000 edges to draw.
        Note that drawing a large number of edges might be particularly slow.
    cp : list of int
        NOT IMPLEMENTED. Camera position when plotting a 3D graph.
    vertex_size : float
        Size of circle representing each node.
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
    plot_name : string
        Name of the plot.
    save_as : str
        Whether to save the plot as save_as.png and save_as.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed. Only
        available with the matplotlib backend.

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Grid2d(4)
    >>> signal = np.sin((np.arange(16) * 2*np.pi/16))
    >>> plotting.plot_signal(G, signal)

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
        raise ValueError("Can't display complex signal.")

    kwargs['show_edges'] = kwargs.pop('show_edges', G.Ne < 10e3)

    default = G.plotting['vertex_size']
    kwargs['vertex_size'] = kwargs.pop('vertex_size', default)

    plot_name = u'{}\nG.N={} nodes, G.Ne={} edges'.format(G.gtype, G.N, G.Ne)
    kwargs['plot_name'] = kwargs.pop('plot_name', plot_name)

    limits = [1.05*signal.min(), 1.05*signal.max()]
    kwargs['limits'] = kwargs.pop('limits', limits)

    if backend is None:
        backend = BACKEND

    G = _handle_directed(G)

    if backend == 'pyqtgraph':
        _qtg_plot_signal(G, signal, **kwargs)
    elif backend == 'matplotlib':
        _plt_plot_signal(G, signal, **kwargs)
    else:
        raise ValueError('Unknown backend {}.'.format(backend))


@_plt_handle_figure
def _plt_plot_signal(G, signal, show_edges, limits, ax,
                     vertex_size, highlight=[], colorbar=True):

    if show_edges:

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


def _qtg_plot_signal(G, signal, show_edges, plot_name, vertex_size, limits):

    qtg, gl, QtGui = _import_qtg()

    if G.coords.shape[1] == 2:
        window = qtg.GraphicsWindow(plot_name)
        view = window.addViewBox()

    elif G.coords.shape[1] == 3:
        if not QtGui.QApplication.instance():
            QtGui.QApplication([])  # We want only one application.
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle(plot_name)

    if show_edges:

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


def plot_spectrogram(G, node_idx=None):
    r"""
    Plot the spectrogram of the given graph.

    Parameters
    ----------
    G : Graph
        Graph to analyse.
    node_idx : ndarray
        Order to sort the nodes in the spectrogram

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Ring(15)
    >>> plotting.plot_spectrogram(G)

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
