# -*- coding: utf-8 -*-

r"""
The :mod:`pygsp.plotting` module implements functionality to plot PyGSP objects
with a `pyqtgraph <http://www.pyqtgraph.org>`_ or `matplotlib
<https://matplotlib.org>`_ drawing backend:

* graphs from :mod:`pygsp.graphs` with :func:`plot_graph`,
  :func:`plot_spectrogram`, and :func:`plot_signal`,
* filters from :mod:`pygsp.filters` with :func:`plot_filter`.

"""

import numpy as np
import uuid

try:
    import matplotlib.pyplot as plt
    plt_import = True
except Exception as e:
    print('ERROR : Could not import packages for matplotlib.')
    print('Details : {}'.format(e))
    plt_import = False

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    import pyqtgraph.opengl as gl
    qtg_import = True
except Exception as e:
    print('ERROR : Could not import packages for pyqtgraph.')
    print('Details : {}'.format(e))
    qtg_import = False


class plid():
    r"""Not so clean way of generating plot_ids."""

    def __init__(self):
        self.plot_id = 0


plid = plid()


def show(block=False, **kwargs):
    r"""
    Show created figures.

    Alias to plt.show().
    By default, showing plots does not block the prompt.

    """
    plt.show(block, **kwargs)


def close(*args, **kwargs):
    r"""
    Close created figures.

    Alias to plt.close().

    """
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
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> plotting.plot(G, default_qtg=False)

    """
    from .graphs import Graph
    from .filters import Filter

    if issubclass(type(O), Graph):
        plot_graph(O, **kwargs)
    elif issubclass(type(O), Filter):
        plot_filter(O, **kwargs)
    else:
        raise TypeError('Unrecognized object type, i.e. not Graph or Filter.')


def plot_graph(G, default_qtg=True, **kwargs):
    r"""
    Plot a graph or a list of graphs.

    This function should be able to determine the appropriate plot for
    the graph.

    Parameters
    ----------
    G : Graph
        Graph to plot.
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    default_qtg: boolean
        define the drawing backend to use if both are available.
        Default True, i.e. pyqtgraph.
    plot_name : string
        name of the plot
    savefig : boolean
        whether the plot is saved as plot_name.png and plot_name.pdf (True) or
        shown in a window (False) (default False). Only available with the
        matplotlib backend.
    show_plot : boolean
        whether to show the plot, i.e. call plt.show(). Only available with the
        matplotlib backend.

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> plotting.plot_graph(G, default_qtg=False)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coords() first.')

    if qtg_import and (default_qtg or not plt_import):
        _pg_plot_graph(G, **kwargs)
    elif plt_import and not (default_qtg and qtg_import):
        _plt_plot_graph(G, **kwargs)
    else:
        raise ImportError('No drawing library installed. Please '
                          'install matplotlib or pyqtgraph.')


def _plt_plot_graph(G, savefig=False, show_edges=None, show_plot=True, plot_name=''):

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    fig = plt.figure(plid.plot_id)
    plid.plot_id += 1

    if not plot_name:
        plot_name = u"Plot of {}".format(G.gtype)

    if show_edges is None:
        show_edges = G.Ne < 10000

    if 'edge_color' not in G.plotting:
        G.plotting['edge_color'] = np.array([255, 88, 41])/255.

    # Matplotlib graph initialization in 2D and 3D
    if G.coords.shape[1] == 2:
        ax = fig.add_subplot(111)
    elif G.coords.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')

    if show_edges:
        ki, kj = np.nonzero(G.A)
        if G.is_directed():
            raise NotImplementedError
        else:
            if G.coords.shape[1] == 2:
                ki, kj = np.nonzero(G.A)
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                    np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                    np.expand_dims(G.coords[kj, 1], axis=0)))

                if isinstance(G.plotting['vertex_color'], list):
                    ax.plot(x, y, linewidth=G.plotting['edge_width'],
                            color=G.plotting['edge_color'],
                            linestyle=G.plotting['edge_style'],
                            marker='', zorder=1)

                    ax.scatter(G.coords[:, 0], G.coords[:, 1], marker='o',
                               s=G.plotting['vertex_size'],
                               c=G.plotting['vertex_color'], zorder=2)
                else:
                    ax.plot(x, y, linewidth=G.plotting['edge_width'],
                            color=G.plotting['edge_color'],
                            linestyle=G.plotting['edge_style'],
                            marker='o', markersize=G.plotting['vertex_size'],
                            markerfacecolor=G.plotting['vertex_color'])

            if G.coords.shape[1] == 3:
                # Very dirty way to display a 3d graph
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                    np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                    np.expand_dims(G.coords[kj, 1], axis=0)))
                z = np.concatenate((np.expand_dims(G.coords[ki, 2], axis=0),
                                    np.expand_dims(G.coords[kj, 2], axis=0)))
                ii = range(0, x.shape[1])
                x2 = np.ndarray((0, 1))
                y2 = np.ndarray((0, 1))
                z2 = np.ndarray((0, 1))
                for i in ii:
                    x2 = np.append(x2, x[:, i])
                for i in ii:
                    y2 = np.append(y2, y[:, i])
                for i in ii:
                    z2 = np.append(z2, z[:, i])
                for i in range(0, x.shape[1] * 2, 2):
                    x3 = x2[i:i + 2]
                    y3 = y2[i:i + 2]
                    z3 = z2[i:i + 2]
                    ax.plot(x3, y3, z3, linewidth=G.plotting['edge_width'],
                            color=G.plotting['edge_color'],
                            linestyle=G.plotting['edge_style'],
                            marker='o', markersize=G.plotting['vertex_size'],
                            markerfacecolor=G.plotting['vertex_color'])
    else:
        # TODO: is ax.plot(G.coords[:, 0], G.coords[:, 1], 'bo') faster?
        if G.coords.shape[1] == 2:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], marker='o',
                       s=G.plotting['vertex_size'],
                       c=G.plotting['vertex_color'])
        if G.coords.shape[1] == 3:
            ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2],
                       marker='o', s=G.plotting['vertex_size'],
                       c=G.plotting['vertex_color'])

    # Save plot as PNG or show it in a window
    if savefig:
        plt.savefig(plot_name + '.png')
        plt.savefig(plot_name + '.pdf')
    elif show_plot:
        plt.show(False)  # non blocking show

    # threading.Thread(None, _thread, None, (G, show_edges, savefig)).start()


def _pg_plot_graph(G, show_edges=None, plot_name=''):

    # TODO handling when G is a list of graphs
    global window_list
    if 'window_list' not in globals():
        window_list = {}

    if show_edges is None:
        show_edges = G.Ne < 10000

    ki, kj = np.nonzero(G.A)
    if G.is_directed():
        raise NotImplementedError
    else:
        if G.coords.shape[1] == 2:
            adj = np.concatenate((np.expand_dims(ki, axis=1),
                                  np.expand_dims(kj, axis=1)), axis=1)

            w = pg.GraphicsWindow()
            w.setWindowTitle(G.plotting['plot_name'] if 'plot_name' in G.plotting else plot_name or G.gtype)
            v = w.addViewBox()
            v.setAspectLocked()

            extra_args = {}
            if isinstance(G.plotting['vertex_color'], list):
                extra_args['symbolPen'] = [pg.mkPen(v_col) for v_col in G.plotting['vertex_color']]
                extra_args['brush'] = [pg.mkBrush(v_col) for v_col in G.plotting['vertex_color']]
            elif isinstance(G.plotting['vertex_color'], int):
                extra_args['symbolPen'] = G.plotting['vertex_color']
                extra_args['brush'] = G.plotting['vertex_color']

            # Define syntaxic sugar mapping keywords for the display options
            for plot_args, pg_args in [('vertex_size', 'size'), ('vertex_mask', 'mask'), ('edge_color', 'pen')]:
                if plot_args in G.plotting:
                    G.plotting[pg_args] = G.plotting.pop(plot_args)

            for pg_args in ['size', 'mask', 'pen', 'symbolPen']:
                if pg_args in G.plotting:
                    extra_args[pg_args] = G.plotting[pg_args]

            if not show_edges:
                extra_args['pen'] = None

            g = pg.GraphItem(pos=G.coords, adj=adj, **extra_args)
            v.addItem(g)

            window_list[str(uuid.uuid4())] = w

        elif G.coords.shape[1] == 3:
            app = QtGui.QApplication([])
            w = gl.GLViewWidget()
            w.opts['distance'] = 10
            w.show()
            w.setWindowTitle(G.plotting['plot_name'] if 'plot_name' in G.plotting else plot_name or G.gtype)

            # Very dirty way to display a 3d graph
            x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                np.expand_dims(G.coords[kj, 0], axis=0)))
            y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                np.expand_dims(G.coords[kj, 1], axis=0)))
            z = np.concatenate((np.expand_dims(G.coords[ki, 2], axis=0),
                                np.expand_dims(G.coords[kj, 2], axis=0)))
            ii = range(0, x.shape[1])
            x2 = np.ndarray((0, 1))
            y2 = np.ndarray((0, 1))
            z2 = np.ndarray((0, 1))
            for i in ii:
                x2 = np.append(x2, x[:, i])
            for i in ii:
                y2 = np.append(y2, y[:, i])
            for i in ii:
                z2 = np.append(z2, z[:, i])

            pts = np.concatenate((np.expand_dims(x2, axis=1),
                                  np.expand_dims(y2, axis=1),
                                  np.expand_dims(z2, axis=1)), axis=1)

            extra_args = {'color': (0, 0, 1, 1)}
            if 'vertex_color' in G.plotting:
                if isinstance(G.plotting['vertex_color'], list):
                    extra_args['color'] = np.array([pg.glColor(pg.mkPen(v_col).color()) for v_col in G.plotting['vertex_color']])
                elif isinstance(G.plotting['vertex_color'], int):
                    extra_args['color'] = pg.glColor(pg.mkPen(G.plotting['vertex_color']).color())
                else:
                    extra_args['color'] = G.plotting['vertex_color']

            # Define syntaxic sugar mapping keywords for the display options
            for plot_args, pg_args in [('vertex_size', 'size')]:
                if plot_args in G.plotting:
                    G.plotting[pg_args] = G.plotting.pop(plot_args)

            for pg_args in ['size']:
                if pg_args in G.plotting:
                    extra_args[pg_args] = G.plotting[pg_args]

            if show_edges:
                g = gl.GLLinePlotItem(pos=pts, mode='lines', color=G.plotting['edge_color'])
                w.addItem(g)

            gp = gl.GLScatterPlotItem(pos=G.coords, **extra_args)
            w.addItem(gp)

            window_list[str(uuid.uuid4())] = app


def plot_filter(filters, npoints=1000, line_width=4, x_width=3,
                x_size=10, plot_eigenvalues=None, show_sum=None,
                savefig=False, show_plot=False, plot_name=None):
    r"""
    Plot a system of graph spectral filters.

    Parameters
    ----------
    filters : Filter
        Filter to plot.
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
    savefig : boolean
        whether the plot is saved as plot_name.png and plot_name.pdf (True) or
        shown in a window (False) (default False). Only available with the
        matplotlib backend.

    Examples
    --------
    >>> from pygsp import filters, plotting, graphs
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> plotting.plot_filter(mh)

    """
    G = filters.G

    if not isinstance(filters.g, list):
        filters.g = [filters.g]
    if plot_eigenvalues is None:
        plot_eigenvalues = hasattr(G, 'e')
    if show_sum is None:
        show_sum = len(filters.g) > 1
    if plot_name is None:
        plot_name = u"Filter plot of {}".format(G.gtype)

    lambdas = np.linspace(0, G.lmax, npoints)

    # Apply the filter
    fd = filters.evaluate(lambdas)

    # Plot the filter
    size = len(fd)
    fig = plt.figure(plid.plot_id)
    plid.plot_id += 1
    ax = fig.add_subplot(111)
    if len(filters.g) == 1:
        ax.plot(lambdas, fd, linewidth=line_width)
    elif len(filters.g) > 1:
        for i in range(size):
            ax.plot(lambdas, fd[i], linewidth=line_width)

    # Plot eigenvalues
    if plot_eigenvalues:
        ax.plot(G.e, np.zeros(G.N), 'xk', markeredgewidth=x_width,
                markersize=x_size)

    # Plot highlighted eigenvalues TODO

    # Plot the sum
    if show_sum:
        test_sum = np.sum(np.power(fd, 2), 0)
        ax.plot(lambdas, test_sum, 'k', linewidth=line_width)

    # Save plot as PNG or show it in a window
    if savefig:
        plt.savefig(plot_name + '.png')
        plt.savefig(plot_name + '.pdf')
    elif show_plot:
        plt.show(False)  # non blocking show


def plot_signal(G, signal, default_qtg=True, **kwargs):
    r"""
    Plot a signal on top of a graph.

    Parameters
    ----------
    G : Graph
        Graph to plot a signal on top.
    signal : array of int
        Signal to plot. Signal length should be equal to the number of nodes.
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    cp : list of int
        Camera position for a 3D graph.
    vertex_size : int
        Size of circle representing each signal component.
    vertex_highlight : list of boolean
        Vector of indices for vertices to be highlighted.
    climits : list of int
        Limits of the colorbar.
    colorbar : boolean
        To plot an extra line showing the sum of the squared magnitudes
        of the filters (default True if there is multiple filters).
    bar : boolean
        NOT IMPLEMENTED: False display color, True display bar for the graph
        (default False).
    bar_width : int
        Width of the bar (default 1).
    default_qtg: boolean
        define the drawing backend to use if both are available.
        Default True, i.e. pyqtgraph.
    plot_name : string
        name of the plot
    savefig : boolean
        whether the plot is saved as plot_name.png and plot_name.pdf (True) or
        shown in a window (False) (default False). Only available with the
        matplotlib backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, filters, plotting
    >>> G = graphs.Grid2d(4)
    >>> signal = np.sin((np.arange(16) * 2*np.pi/16))
    >>> plotting.plot_signal(G, signal, default_qtg=False)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coords() first.')

    if qtg_import and (default_qtg or not plt_import):
        _pg_plot_signal(G, signal, **kwargs)
    elif plt_import and not (default_qtg and qtg_import):
        _plt_plot_signal(G, signal, **kwargs)
    else:
        raise ImportError('No drawing library installed. Please '
                          'install matplotlib or pyqtgraph.')


def _plt_plot_signal(G, signal, show_edges=None, cp=[-6, -3, 160],
                     vertex_size=None, vertex_highlight=False, climits=None,
                     colorbar=True, bar=False, bar_width=1, savefig=False,
                     show_plot=False, plot_name=None):

    fig = plt.figure(plid.plot_id)
    plid.plot_id += 1

    if np.sum(np.abs(signal.imag)) > 1e-10:
        raise ValueError("Can't display complex signal.")
    if show_edges is None:
        show_edges = G.Ne < 10000
    if vertex_size is None:
        vertex_size = 100
    if climits is None:
        cmin = 1.01 * np.min(signal)
        cmax = 1.01 * np.max(signal)
        climits = [cmin, cmax]
    if plot_name is None:
        plot_name = "Signal plot of " + G.gtype

    # Matplotlib graph initialization in 2D and 3D
    if G.coords.shape[1] == 2:
        ax = fig.add_subplot(111)
    elif G.coords.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    if show_edges:
        ki, kj = np.nonzero(G.A)

        if G.is_directed():
            raise NotImplementedError

        else:
            if G.coords.shape[1] == 2:
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                    np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                    np.expand_dims(G.coords[kj, 1], axis=0)))
                ax.plot(x, y, color='grey', zorder=1)
                # plt.show()
            if G.coords.shape[1] == 3:
                # Very dirty way to display 3D graph edges
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                    np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                    np.expand_dims(G.coords[kj, 1], axis=0)))
                z = np.concatenate((np.expand_dims(G.coords[ki, 2], axis=0),
                                    np.expand_dims(G.coords[kj, 2], axis=0)))
                ii = range(0, x.shape[1])
                x2 = np.ndarray((0, 1))
                y2 = np.ndarray((0, 1))
                z2 = np.ndarray((0, 1))
                for i in ii:
                    x2 = np.append(x2, x[:, i])
                for i in ii:
                    y2 = np.append(y2, y[:, i])
                for i in ii:
                    z2 = np.append(z2, z[:, i])
                for i in range(0, x.shape[1] * 2, 2):
                    x3 = x2[i:i + 2]
                    y3 = y2[i:i + 2]
                    z3 = z2[i:i + 2]
                    ax.plot(x3, y3, z3, color='grey', marker='o',
                            markerfacecolor='blue', zorder=1)

    # Plot signal
    if G.coords.shape[1] == 2:
        ax.scatter(G.coords[:, 0], G.coords[:, 1], s=vertex_size, c=signal,
                   zorder=2)
    if G.coords.shape[1] == 3:
        ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2],
                   s=vertex_size, c=signal, zorder=2)

    # Save plot as PNG or show it in a window
    if savefig:
        plt.savefig(plot_name + '.png')
        plt.savefig(plot_name + '.pdf')
    elif show_plot:
        plt.show(False)  # non blocking show


def _pg_plot_signal(G, signal, show_edges=None, cp=[-6, -3, 160],
                    vertex_size=None, vertex_highlight=False, climits=None,
                    colorbar=True, bar=False, bar_width=1, plot_name=None):

    if np.sum(np.abs(signal.imag)) > 1e-10:
        raise ValueError("Can't display complex signal.")

    if show_edges is None:
        show_edges = G.Ne < 10000
    if vertex_size is None:
        vertex_size = 15
    if climits is None:
        cmin = 1.01 * np.min(signal)
        cmax = 1.01 * np.max(signal)
        climits = [cmin, cmax]

    # pygtgraph window initialization in 2D and 3D
    global window_list
    if 'window_list' not in globals():
        window_list = {}

    if G.coords.shape[1] == 2:
        w = pg.GraphicsWindow(plot_name or G.gtype)
        v = w.addViewBox()
    elif G.coords.shape[1] == 3:
        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.opts['distance'] = 10
        w.show()
        w.setWindowTitle(plot_name or G.gtype)

    # Plot edges
    if show_edges:
        ki, kj = np.nonzero(G.A)
        if G.is_directed():
            raise NotImplementedError
        else:
            if G.coords.shape[1] == 2:
                adj = np.concatenate((np.expand_dims(ki, axis=1),
                                      np.expand_dims(kj, axis=1)), axis=1)

                g = pg.GraphItem(pos=G.coords, adj=adj, symbolBrush=None,
                                 symbolPen=None)
                v.addItem(g)

            if G.coords.shape[1] == 3:
                # Very dirty way to display a 3d graph
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0),
                                    np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0),
                                    np.expand_dims(G.coords[kj, 1], axis=0)))
                z = np.concatenate((np.expand_dims(G.coords[ki, 2], axis=0),
                                    np.expand_dims(G.coords[kj, 2], axis=0)))
                ii = range(0, x.shape[1])
                x2 = np.ndarray((0, 1))
                y2 = np.ndarray((0, 1))
                z2 = np.ndarray((0, 1))
                for i in ii:
                    x2 = np.append(x2, x[:, i])
                for i in ii:
                    y2 = np.append(y2, y[:, i])
                for i in ii:
                    z2 = np.append(z2, z[:, i])

                pts = np.concatenate((np.expand_dims(x2, axis=1),
                                      np.expand_dims(y2, axis=1),
                                      np.expand_dims(z2, axis=1)), axis=1)

                g = gl.GLLinePlotItem(pos=pts, mode='lines')

                gp = gl.GLScatterPlotItem(pos=G.coords, color=(1., 0., 0., 1))

                w.addItem(g)
                w.addItem(gp)

    # Plot signal on top
    pos = [1, 8, 24, 40, 56, 64]
    color = np.array([[0, 0, 143, 255], [0, 0, 255, 255], [0, 255, 255, 255],
                      [255, 255, 0, 255], [255, 0, 0, 255], [128, 0, 0, 255]])
    cmap = pg.ColorMap(pos, color)

    mininum = min(signal)
    maximum = max(signal)

    normalized_signal = [1 + 63 *(float(x) - mininum) / (maximum - mininum) for x in signal]

    if G.coords.shape[1] == 2:
        gp = pg.ScatterPlotItem(G.coords[:, 0],
                                G.coords[:, 1],
                                size=vertex_size,
                                brush=cmap.map(normalized_signal, 'qcolor'))
        v.addItem(gp)
    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(G.coords[:, 0], G.coords[:, 1],
                                  G.coords[:, 2], size=vertex_size, c=signal)
        w.addItem(gp)


    # Multiple windows handling
    if G.coords.shape[1] == 2:
        window_list[str(uuid.uuid4())] = w
    elif G.coords.shape[1] == 3:
        window_list[str(uuid.uuid4())] = app


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
    >>> import numpy as np
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Ring(15)
    >>> plotting.plot_spectrogram(G)

    """
    global window_list
    from pygsp.features import compute_spectrogram
    if 'window_list' not in globals():
        window_list = {}

    if not qtg_import:
        raise NotImplementedError("You need pyqtgraph to plot the spectrogram at the moment. Please install dependency and retry.")

    if not hasattr(G, 'spectr'):
        compute_spectrogram(G)

    M = G.spectr.shape[1]
    spectr = np.ravel(G.spectr[node_idx, :] if node_idx is not None else G.spectr)
    min_spec, max_spec = np.min(spectr), np.max(spectr)

    pos = np.array([0., 0.25, 0.5, 0.75, 1.])
    color = np.array([[20, 133, 212, 255], [53, 42, 135, 255], [48, 174, 170, 255],
                     [210, 184, 87, 255], [249, 251, 14, 255]], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)

    w = pg.GraphicsWindow()
    w.setWindowTitle("Spectrogramm of {}".format(G.gtype))
    v = w.addPlot(labels={'bottom': 'nodes',
                          'left': 'frequencies {}:{:.2f}:{:.2f}'.format(0, G.lmax/M, G.lmax)})
    v.setAspectLocked()

    spi = pg.ScatterPlotItem(np.repeat(np.arange(G.N), M), np.ravel(np.tile(np.arange(M), (1, G.N))), pxMode=False, symbol='s',
                             size=1, brush=cmap.map((spectr.astype(float) - min_spec)/(max_spec - min_spec), 'qcolor'))
    v.addItem(spi)

    window_list[str(uuid.uuid4())] = w
