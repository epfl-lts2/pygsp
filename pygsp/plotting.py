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

import numpy as np

try:
    import matplotlib.pyplot as plt
    # Not used directly, but needed for 3D projection.
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    plt_import = True
except Exception as e:
    print('ERROR : Could not import packages for matplotlib.')
    print('Details : {}'.format(e))
    plt_import = False

try:
    import pyqtgraph as qtg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtGui
    qtg_import = True
except Exception as e:
    print('ERROR : Could not import packages for pyqtgraph.')
    print('Details : {}'.format(e))
    qtg_import = False


BACKEND = 'pyqtgraph'
_qtg_windows = []
_qtg_widgets = []
_plt_figures = []


def _plt_handle_figure(plot):

    def inner(obj, *args, **kwargs):

        # Create a figure and an axis if none were passed.
        if not 'ax' in kwargs.keys():
            fig = plt.figure()
            global _plt_figures
            _plt_figures.append(fig)

            if hasattr(obj, 'coords') and obj.coords.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            kwargs.update(ax=ax)

        save_as = kwargs.pop('save_as', None)

        plot(obj, *args, **kwargs)

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
        plt.close(fig)
    _plt_figures = []


def show(*args, **kwargs):
    r"""
    Show created figures.

    Alias to plt.show().
    By default, showing plots does not block the prompt.

    """
    plt.show(*args, **kwargs)


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
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    backend: {'matplotlib', 'pyqtgraph'}
        Defines the drawing backend to use. Defaults to :data:`BACKEND`.
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
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> plotting.plot_graph(G)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')
    if G.coords.shape[1] not in [2, 3]:
        raise AttributeError('Coordinates should be in 2D or 3D space.')

    if backend is None:
        backend = BACKEND

    if backend == 'pyqtgraph' and qtg_import:
        _qtg_plot_graph(G, **kwargs)
    elif backend == 'matplotlib' and plt_import:
        _plt_plot_graph(G, **kwargs)
    else:
        raise ValueError('The {} backend is not available.'.format(backend))


@_plt_handle_figure
def _plt_plot_graph(G, show_edges=None, plot_name='', ax=None):

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    if not plot_name:
        plot_name = u"Plot of {}".format(G.gtype)

    if show_edges is None:
        show_edges = G.Ne < 10000

    try:
        vertex_size = G.plotting['vertex_size']
    except KeyError:
        vertex_size = 100

    try:
        edge_color = G.plotting['edge_color']
    except KeyError:
        edge_color = [0.5, 0.5, 0.5]

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
                            color=edge_color,
                            linestyle=G.plotting['edge_style'],
                            marker='', zorder=1)

                    ax.scatter(G.coords[:, 0], G.coords[:, 1], marker='o',
                               s=vertex_size,
                               c=G.plotting['vertex_color'], zorder=2)
                else:
                    ax.plot(x, y, linewidth=G.plotting['edge_width'],
                            color=edge_color,
                            linestyle=G.plotting['edge_style'],
                            marker='o', markersize=vertex_size,
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
                            color=edge_color,
                            linestyle=G.plotting['edge_style'],
                            marker='o', markersize=vertex_size,
                            markerfacecolor=G.plotting['vertex_color'])
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

    # threading.Thread(None, _thread, None, (G, show_edges, savefig)).start()


def _qtg_plot_graph(G, show_edges=None, plot_name=''):

    # TODO handling when G is a list of graphs

    if show_edges is None:
        show_edges = G.Ne < 10000

    try:
        edge_color = G.plotting['edge_color']
    except KeyError:
        edge_color = [0.5, 0.5, 0.5]

    ki, kj = np.nonzero(G.A)
    if G.is_directed():
        raise NotImplementedError
    else:
        if G.coords.shape[1] == 2:
            adj = np.concatenate((np.expand_dims(ki, axis=1),
                                  np.expand_dims(kj, axis=1)), axis=1)

            window = qtg.GraphicsWindow()
            window.setWindowTitle(G.plotting['plot_name'] if 'plot_name' in G.plotting else plot_name or G.gtype)
            view = window.addViewBox()
            view.setAspectLocked()

            extra_args = {}
            if isinstance(G.plotting['vertex_color'], list):
                extra_args['symbolPen'] = [qtg.mkPen(v_col) for v_col in G.plotting['vertex_color']]
                extra_args['brush'] = [qtg.mkBrush(v_col) for v_col in G.plotting['vertex_color']]
            elif isinstance(G.plotting['vertex_color'], int):
                extra_args['symbolPen'] = G.plotting['vertex_color']
                extra_args['brush'] = G.plotting['vertex_color']

            # Define syntactic sugar mapping keywords for the display options
            for plot_args, qtg_args in [('vertex_size', 'size'), ('vertex_mask', 'mask'), ('edge_color', 'pen'), ('symbolPen', 'symbolPen')]:
                if plot_args in G.plotting:
                    extra_args[qtg_args] = G.plotting[plot_args]

            if not show_edges:
                extra_args['pen'] = None

            g = qtg.GraphItem(pos=G.coords, adj=adj, **extra_args)
            view.addItem(g)

            global _qtg_windows
            _qtg_windows.append(window)

        elif G.coords.shape[1] == 3:
            if not QtGui.QApplication.instance():
                # We want only one application.
                QtGui.QApplication([])
            widget = gl.GLViewWidget()
            widget.opts['distance'] = 10
            widget.show()
            widget.setWindowTitle(G.plotting['plot_name'] if 'plot_name' in G.plotting else plot_name or G.gtype)

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
                    extra_args['color'] = np.array([qtg.glColor(qtg.mkPen(v_col).color()) for v_col in G.plotting['vertex_color']])
                elif isinstance(G.plotting['vertex_color'], int):
                    extra_args['color'] = qtg.glColor(qtg.mkPen(G.plotting['vertex_color']).color())
                else:
                    extra_args['color'] = G.plotting['vertex_color']

            # Define syntaxic sugar mapping keywords for the display options
            for plot_args, qtg_args in [('vertex_size', 'size')]:
                if plot_args in G.plotting:
                    G.plotting[qtg_args] = G.plotting.pop(plot_args)

            for qtg_args in ['size']:
                if qtg_args in G.plotting:
                    extra_args[qtg_args] = G.plotting[qtg_args]

            if show_edges:
                g = gl.GLLinePlotItem(pos=pts, mode='lines', color=edge_color)
                widget.addItem(g)

            gp = gl.GLScatterPlotItem(pos=G.coords, **extra_args)
            widget.addItem(gp)

            global _qtg_widgets
            _qtg_widgets.append(widget)


@_plt_handle_figure
def plot_filter(filters, npoints=1000, line_width=4, x_width=3,
                x_size=10, plot_eigenvalues=None, show_sum=None,
                plot_name=None, ax=None):
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
    >>> from pygsp import graphs, filters, plotting
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> plotting.plot_filter(mh)

    """

    G = filters.G

    if plot_eigenvalues is None:
        plot_eigenvalues = hasattr(G, '_e')
    if show_sum is None:
        show_sum = filters.Nf > 1
    if plot_name is None:
        plot_name = u"Filter plot of {}".format(G.gtype)

    lambdas = np.linspace(0, G.lmax, npoints)

    # Apply the filter
    fd = filters.evaluate(lambdas)

    # Plot the filter
    if filters.Nf == 1:
        ax.plot(lambdas, fd, linewidth=line_width)
    else:
        for fd_i in fd:
            ax.plot(lambdas, fd_i, linewidth=line_width)

    # Plot eigenvalues
    if plot_eigenvalues:
        ax.plot(G.e, np.zeros(G.N), 'xk', markeredgewidth=x_width,
                markersize=x_size)

    # TODO: plot highlighted eigenvalues

    # Plot the sum
    if show_sum:
        test_sum = np.sum(np.power(fd, 2), 0)
        ax.plot(lambdas, test_sum, 'k', linewidth=line_width)


def plot_signal(G, signal, backend=None, **kwargs):
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
        NOT IMPLEMENTED. Camera position when plotting a 3D graph.
    vertex_size : int
        Size of circle representing each signal component.
    vertex_highlight : list of boolean
        NOT IMPLEMENTED. Vector of indices for vertices to be highlighted.
    colorbar : bool
        Whether to plot a colorbar indicating the signal's amplitude.
        Only available with the matplotlib backend.
    limits : [vmin, vmax]
        Maps colors from vmin to vmax.
        Defaults to signal minimum and maximum value.
        Only available with the matplotlib backend.
    bar : boolean
        NOT IMPLEMENTED: False display color, True display bar for the graph
        (default False).
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
    >>> import numpy as np
    >>> from pygsp import graphs, filters, plotting
    >>> G = graphs.Grid2d(4)
    >>> signal = np.sin((np.arange(16) * 2*np.pi/16))
    >>> plotting.plot_signal(G, signal)

    """
    if not hasattr(G, 'coords'):
        raise AttributeError('Graph has no coordinate set. '
                             'Please run G.set_coordinates() first.')

    if backend is None:
        backend = BACKEND

    if backend == 'pyqtgraph' and qtg_import:
        _qtg_plot_signal(G, signal, **kwargs)
    elif backend == 'matplotlib' and plt_import:
        _plt_plot_signal(G, signal, **kwargs)
    else:
        raise ValueError('The {} backend is not available.'.format(backend))


@_plt_handle_figure
def _plt_plot_signal(G, signal, show_edges=None, vertex_size=100, limits=None,
                     colorbar=True, plot_name=None, ax=None):

    if np.sum(np.abs(signal.imag)) > 1e-10:
        raise ValueError("Can't display complex signal.")
    if show_edges is None:
        show_edges = G.Ne < 10000
    if plot_name is None:
        plot_name = "Signal plot of " + G.gtype
    if limits is None:
        limits = [signal.min(), signal.max()]

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
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1],
                        s=vertex_size, c=signal, zorder=2,
                        vmin=limits[0], vmax=limits[1])
    if G.coords.shape[1] == 3:
        sc = ax.scatter(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2],
                        s=vertex_size, c=signal, zorder=2,
                        vmin=limits[0], vmax=limits[1])
        try:
            ax.view_init(elev=G.plotting['elevation'],
                         azim=G.plotting['azimuth'])
            ax.dist = G.plotting['distance']
        except KeyError:
            pass

    if colorbar:
        plt.colorbar(sc, ax=ax)


def _qtg_plot_signal(G, signal, show_edges=None,
                     vertex_size=15, plot_name=None):

    if np.sum(np.abs(signal.imag)) > 1e-10:
        raise ValueError("Can't display complex signal.")

    if show_edges is None:
        show_edges = G.Ne < 10000

    if G.coords.shape[1] == 2:
        window = qtg.GraphicsWindow(plot_name or G.gtype)
        view = window.addViewBox()
    elif G.coords.shape[1] == 3:
        if not QtGui.QApplication.instance():
            # We want only one application.
            QtGui.QApplication([])
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 10
        widget.show()
        widget.setWindowTitle(plot_name or G.gtype)

    # Plot edges
    if show_edges:
        ki, kj = np.nonzero(G.A)
        if G.is_directed():
            raise NotImplementedError
        else:
            if G.coords.shape[1] == 2:
                adj = np.concatenate((np.expand_dims(ki, axis=1),
                                      np.expand_dims(kj, axis=1)), axis=1)

                g = qtg.GraphItem(pos=G.coords, adj=adj, symbolBrush=None,
                                 symbolPen=None)
                view.addItem(g)

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

                widget.addItem(g)
                widget.addItem(gp)

    # Plot signal on top
    pos = [1, 8, 24, 40, 56, 64]
    color = np.array([[0, 0, 143, 255], [0, 0, 255, 255], [0, 255, 255, 255],
                      [255, 255, 0, 255], [255, 0, 0, 255], [128, 0, 0, 255]])
    cmap = qtg.ColorMap(pos, color)

    mininum = min(signal)
    maximum = max(signal)

    normalized_signal = [1 + 63 *(float(x) - mininum) / (maximum - mininum) for x in signal]

    if G.coords.shape[1] == 2:
        gp = qtg.ScatterPlotItem(G.coords[:, 0],
                                G.coords[:, 1],
                                size=vertex_size,
                                brush=cmap.map(normalized_signal, 'qcolor'))
        view.addItem(gp)
    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(pos=G.coords, size=vertex_size, color=signal)
        widget.addItem(gp)

    # Multiple windows handling
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
    >>> import numpy as np
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Ring(15)
    >>> plotting.plot_spectrogram(G)

    """
    from pygsp import features

    if not qtg_import:
        raise NotImplementedError("You need pyqtgraph to plot the spectrogram at the moment. Please install dependency and retry.")

    if not hasattr(G, 'spectr'):
        features.compute_spectrogram(G)

    M = G.spectr.shape[1]
    spectr = np.ravel(G.spectr[node_idx, :] if node_idx is not None else G.spectr)
    min_spec, max_spec = np.min(spectr), np.max(spectr)

    pos = np.array([0., 0.25, 0.5, 0.75, 1.])
    color = np.array([[20, 133, 212, 255], [53, 42, 135, 255], [48, 174, 170, 255],
                     [210, 184, 87, 255], [249, 251, 14, 255]], dtype=np.ubyte)
    cmap = qtg.ColorMap(pos, color)

    w = qtg.GraphicsWindow()
    w.setWindowTitle("Spectrogramm of {}".format(G.gtype))
    v = w.addPlot(labels={'bottom': 'nodes',
                          'left': 'frequencies {}:{:.2f}:{:.2f}'.format(0, G.lmax/M, G.lmax)})
    v.setAspectLocked()

    spi = qtg.ScatterPlotItem(np.repeat(np.arange(G.N), M), np.ravel(np.tile(np.arange(M), (1, G.N))), pxMode=False, symbol='s',
                             size=1, brush=cmap.map((spectr.astype(float) - min_spec)/(max_spec - min_spec), 'qcolor'))
    v.addItem(spi)

    global _qtg_windows
    _qtg_windows.append(w)
