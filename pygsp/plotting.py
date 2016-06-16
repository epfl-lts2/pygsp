# -*- coding: utf-8 -*-
r"""This module implements plotting functions for the PyGSP main objects."""

import numpy as np
import uuid

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt_import = True
except:
    plt_import = False

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    qtg_import = True
except:
    qtg_import = False


class plid():
    r"""Not so clean way of generating plot_ids."""

    def __init__(self):
        self.plot_id = 0


plid = plid()


def show(block=False):
    r"""
    Show created figures.

    Equivalent to plt.show(*args, **kw) excepted you don't have to import
    matplotlib by youself.

    By default, showing plots does not block the prompt.

    """
    plt.show(block)


def close(*args):
    r"""
    Close created figures.

    Strictly equivalent to plt.close(*args) excepted you don't have to import
    matplotlib by youself.

    By default, showing plots does not block the prompt.

    """
    plt.close(*args)


def plot(O, default_qtg=True, **kwargs):
    r"""
    Main plotting function.

    This function should be able to determine the appropriate plot for
    the object.
    Additionnal kwargs may be given in case of filter plotting.

    Parameters
    ----------
    O : object
        Should be either a Graph, Filter or PointCloud
    default_qtg: boolean
        Define the library to use if both are installed.
        Default is pyqtgraph (field=True).

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> try:
    ...     plotting.plot(G, default_qtg=False)
    ... except Exception as e:
    ...     print(e)

    """
    from .graphs import Graph
    from .pointsclouds.pointscloud import PointsCloud
    from .filters import Filter

    if issubclass(type(O), Graph):
        plot_graph(O, default_qtg, **kwargs)
    elif issubclass(type(O), PointsCloud):
        plot_pointcloud(O)
    elif issubclass(type(O), Filter):
        plot_filter(O, **kwargs)
    else:
        raise TypeError('Your object type is incorrect, be sure it is a '
                        'PointCloud, a Filter or a Graph.')


def plot_graph(G, default_qtg=True, **kwargs):
    r"""
    Plot a graph or an array of graphs with installed libraries.

    This function should be able to determine the appropriate plot for
    the graph.
    Additionnal kwargs may be given in case of filter plotting.

    Parameters
    ----------
    G : Graph
        Graph object to plot
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    default_qtg: boolean
        Define the library to use if both are installed.
        Default is pyqtgraph (field=True).

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> try:
    ...     plotting.plot_graph(G, default_qtg=False)
    ... except Exception as e:
    ...     print(e)

    """
    if qtg_import and (default_qtg or not plt_import):
        pg_plot_graph(G, **kwargs)
    elif plt_import and not (default_qtg and qtg_import):
        plt_plot_graph(G, **kwargs)
    else:
        raise ImportError('No drawing library installed. Please '
                          'install matplotlib or pyqtgraph.')


def plt_plot_graph(G, savefig=False, show_edges=None, plot_name=''):
    r"""
    Plot a graph or an array of graphs with matplotlib.

    See plot_graph for full documentation.

    Extra args
    ----------
    savefig : boolean
        Determine wether the plot is saved as a PNG file in your\
        current directory (True) or shown in a window (False) (default False).
    plot_name : str
        To give custom names to plots

    """
    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    if not plot_name:
        plot_name = u"Plot of {}".format(G.gtype)

    if show_edges is None:
        show_edges = G.Ne < 10000

    if 'edge_color' not in G.plotting:
        G.plotting['edge_color'] = np.array([255, 88, 41])/255.

    if not hasattr(G, 'coords'):
        raise AttributeError('G has no coordinate set. Please run G.set_coords() first.')

    # Matplotlib graph initialization in 2D and 3D
    if G.coords.shape[1] == 2:
        fig = plt.figure(plid.plot_id)
        plid.plot_id += 1
        ax = fig.add_subplot(111)
    elif G.coords.shape[1] == 3:
        fig = plt.figure(plid.plot_id)
        plid.plot_id += 1
        ax = fig.add_subplot(111, projection='3d')

    if show_edges:
        ki, kj = np.nonzero(G.A)
        if G.directed:
            raise NotImplementedError('TODO')
            if G.coords.shape[1] == 2:
                raise NotImplementedError('TODO')
            else:
                raise NotImplementedError('TODO')
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
        # plt.close(fig)
    # else:
    #     plt.show()

    # threading.Thread(None, _thread, None, (G, show_edges, savefig)).start()


def pg_plot_graph(G, show_edges=None):
    r"""
    Plot a graph or an array of graphs.

    See plot_graph for full documentation.

    """
    # TODO handling when G is a list of graphs
    global window_list
    if 'window_list' not in globals():
        window_list = {}

    if not G.coords.shape:
        raise AttributeError('G has no coordinate set. Please run G.set_coords() first.')


    if show_edges is None:
        show_edges = G.Ne < 10000

    ki, kj = np.nonzero(G.A)
    if G.directed:
        raise NotImplementedError('TODO')
        if G.coords.shape[1] == 2:
            raise NotImplementedError('TODO')
        else:
            raise NotImplementedError('TODO')
    else:
        if G.coords.shape[1] == 2:
            adj = np.concatenate((np.expand_dims(ki, axis=1),
                                  np.expand_dims(kj, axis=1)), axis=1)

            w = pg.GraphicsWindow()
            w.setWindowTitle(G.plotting['plot_name'] or G.gtype if 'plot_name' in G.plotting else G.gtype)
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
            w.setWindowTitle(G.plotting['plot_name'] or G.gtype if 'plot_name' in G.plotting else G.gtype)

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


def plot_pointcloud(P):
    r"""
    Plot the coordinates of a pointcloud.

    Parameters
    ----------
    P : PointsClouds object

    Examples
    --------
    >>> from pygsp import plotting, pointsclouds
    >>> logo = pointsclouds.PointsCloud('logo')
    >>> try:
    ...     plotting.plot_pointcloud(logo)
    ... except:
    ...     pass

    """
    if P.coords.shape[1] == 2:
        fig = plt.figure(plid.plot_id)
        plid.plot_id += 1
        ax = fig.add_subplot(111)
        ax.plot(P.coords[:, 0], P.coords[:, 1], 'bo')
        # plt.show()
    if P.coords.shape[1] == 3:
        fig = plt.figure(plid.plot_id)
        plid.plot_id += 1
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(P.coords[:, 0], P.coords[:, 1], P.coords[:, 2], 'bo')
        # plt.show()


def plot_filter(filters, npoints=1000, line_width=4, x_width=3,
                x_size=10, plot_eigenvalues=None, show_sum=None,
                savefig=False, plot_name=None):
    r"""
    Plot a system of graph spectral filters.

    Parameters
    ----------
    filters : filter object
    npoints : int
        Number of point where the filters are evaluated.
    line_width : int
        Width of the filters plots.
    x_width : int
        Width of the X marks representing the eigenvalues.
    x_size : int
        Size of the X marks representing the eigenvalues.
    plot_eigenvalues : boolean
        To plot black X marks at all eigenvalues of the graph (You need to \
        compute the Fourier basis to use this option). By default the \
        eigenvalues are plot if they are contained in the Graph.
    show_sum : boolean
        To plot an extra line showing the sum of the squared magnitudes\
        of the filters (default True if there is multiple filters).
    savefig : boolean
        Determine wether the plot is saved as a PNG file in your\
        current directory (True) or shown in a window (False) (default False).
    plot_name : str
        To give custom names to plots

    Examples
    --------
    >>> from pygsp import filters, plotting, graphs
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> try:
    ...     plotting.plot_filter(mh)
    ... except:
    ...     pass

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
        # plt.close(fig)
    # else:
    #     plt.show()


def plot_signal(G, signal, default_qtg=True, **kwargs):
    r"""
    Plot a graph signal in 2D or 3D with installed libraries.

    Parameters
    ----------
    G : Graph object
        If not specified it will take the one used to create the filter.
    signal : array of int
        Signal applied to the graph.
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    cp : List of int
        Camera position for a 3D graph.
    vertex_size : int
        Size of circle representing each signal component.
    vertex_highlight : boolean
        Vector of indices of vertices to be highlighted.
    climits : array of int
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
        Define the library to use if both are installed.
        Default is pyqtgraph (field=True).

    Examples
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, filters, plotting
    >>> G = graphs.Ring(15)
    >>> signal = np.sin((np.arange(1, 16)*2*np.pi/15))
    >>> try:
    ...     plotting.plot_signal(G, signal, default_qtg=False)
    ... except:
    ...     pass

    """
    if qtg_import and (default_qtg or not plt_import):
        pg_plot_signal(G, signal, **kwargs)
    elif plt_import and not (default_qtg and qtg_import):
        plt_plot_signal(G, signal, **kwargs)
    else:
        raise ImportError('No drawing library installed. Please '
                          'install matplotlib or pyqtgraph.')


def plt_plot_signal(G, signal, show_edges=None, cp=[-6, -3, 160],
                    vertex_size=None, vertex_highlight=False, climits=None,
                    colorbar=True, bar=False, bar_width=1, savefig=False,
                    plot_name=None):
    r"""
    Plot a graph signal in 2D or 3D using matplotlib.

    See plot_signal for full documentation.

    Extra args
    ----------
    savefig : boolean
        Determine whether the plot is saved as a PNG file in your
        current directory (True) or shown in a window (False) (default False).
    plot_name : str
        To give custom names to plots

    """
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
    elif G.coords.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    if show_edges:
        ki, kj = np.nonzero(G.A)

        if G.directed:
            raise NotImplementedError('TODO')
            if G.coords.shape[1] == 2:
                raise NotImplementedError('TODO')
            else:
                raise NotImplementedError('TODO')

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
        # plt.close(fig)
    # else:
    #     plt.show()


def pg_plot_signal(G, signal, show_edges=None, cp=[-6, -3, 160],
                   vertex_size=None, vertex_highlight=False, climits=None,
                   colorbar=True, bar=False, bar_width=1, plot_name=None):
    r"""
    Plot a graph signal in 2D or 3D, with pyqtgraph.

    See plot_signal for full documentation.

    """
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
        if G.directed:
            raise NotImplementedError('TODO')
            if G.coords.shape[1] == 2:
                raise NotImplementedError('TODO')
            else:
                raise NotImplementedError('TODO')
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
    pos = np.arange(0, 1.01, .25)
    color = np.array([[249, 251, 14, 255], [20, 133, 212, 255], [48, 174, 170, 255],
                      [210, 184, 87, 255], [53, 42, 135, 255]])
    cmap = pg.ColorMap(pos, color)

    mininum = min(signal)
    maximum = max(signal)

    normalized_signal = [(float(x) - mininum) / (maximum - mininum) for x in signal]

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


def plot_spectrogramm(G, **kwargs):
    r"""
    Plot the spectrogramm of the given graph.

    Parameters
    ----------
    G : Graph object
        Graph to analyse.
    node_idx : ndarray
        Order to sort the nodes in the spectrogramm

    Example
    --------
    >>> import numpy as np
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Ring(15)
    >>> plotting.plot_spectrogramm(G)

    """
    global window_list
    from pygsp.features import compute_spectrogramm
    if 'window_list' not in globals():
        window_list = {}

    if not qtg_import:
        raise NotImplementedError("You need pyqtgraph to plot the spectrogramm at the moment. Please install dependency and retry.")

    if not hasattr(G, 'spectr'):
        compute_spectrogramm(G)

    M = G.spectr.shape[1]
    node_idx = kwargs.pop('node_idx', None)
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
