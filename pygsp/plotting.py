# -*- coding: utf-8 -*-
r"""
This module implements plotting functions for the pygsp main objects
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui
except:
    pass
import pygsp
import uuid


class plid():
    r"""
    Not so clean way of generating plot_ids
    """

    def __init__(self):
        self.plot_id = 0


plid = plid()


def show():
    r"""
    To show created figures

    Strictly equivalent to plt.show() excepted you don't have to import
    matplotlib by youself.

    """
    plt.show()


def plot(O, **kwargs):
    r"""
    Main plotting function

    This function should be able to determine the appropriated plot for
    the object
    Additionnal kwargs may be given in case of filter plotting

    Parameters
    ----------
    O : object
        Should be either a Graph, Filter or PointCloud

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> G = graphs.Logo()
    >>> try:
    ...     plotting.plot(G)
    ... except:
    ...     pass

    """

    if issubclass(type(O), pygsp.graphs.Graph):
        plot_graph(O)
    elif issubclass(type(O), pygsp.graphs.PointsCloud):
        plot_pointcloud(O)
    elif issubclass(type(O), pygsp.filters.Filter):
        plot_filter(O, **kwargs)
    else:
        raise TypeError('Your object type is incorrect, be sure it is a '
                        'PointCloud, a Filter or a Graph')


def plot_graph(G, savefig=False, show_edges=None, plot_name=False):
    r"""
    Function to plot a graph or an array of graphs

    Parameters
    ----------
    G : Graph
        Graph object to plot
    show_edges : boolean
        Set to False to only draw the vertices (default G.Ne < 10000).
    savefig : boolean
        Determine wether the plot is saved as a PNG file in your\
         current directory (True) or shown in a window (False) (default False).
    plot_name : str
        To give custom names to plots

    Examples
    --------

    >>> from pygsp import plotting, graphs
    >>> G = graphs.Logo()
    >>> try:
    ...     plotting.plot_graph(G)
    ... except:
    ...     pass

    """

    # def _thread(G, show_edges, savefig):

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    if not plot_name:
        plot_name = "Plot of " + G.gtype

    if show_edges is None:
        show_edges = G.Ne < 10000

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
                # ax.plot(x, y, color=G.plotting['edge_color'], marker='o', markerfacecolor=G.plotting['vertex_color'])
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
                    # ax.plot(x3, y3, z3, color=G.plotting['edge_color'], marker='o', markerfacecolor=G.plotting['vertex_color'])
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
        plt.close(fig)
    # else:
    #     plt.show()

    # threading.Thread(None, _thread, None, (G, show_edges, savefig)).start()


def pg_plot_graph(G, show_edges=None):
    r"""
    Function to plot a graph or an array of graphs

    Parameters
    ----------
    G : Graph
        Graph object to plot

    Examples
    --------
    >>> from pygsp import plotting, graphs
    >>> G = graphs.Logo()
    >>> try:
    ...     plotting.pg_plot_graph(G)
    ... except:
    ...     pass


    """

    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph
    global window_list
    if 'window_list' not in globals():
        window_list = {}

    if show_edges is None:
        show_edges = G.Ne < 10000

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
                w = pg.GraphicsWindow()
                w.setWindowTitle(G.gtype)
                v = w.addViewBox()
                v.setAspectLocked()

                g = pg.GraphItem(pos=G.coords, adj=adj)
                v.addItem(g)

                window_list[str(uuid.uuid4())] = w

            if G.coords.shape[1] == 3:
                app = QtGui.QApplication([])
                w = gl.GLViewWidget()
                w.opts['distance'] = 10
                w.show()
                w.setWindowTitle(G.gtype)

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

                window_list[str(uuid.uuid4())] = app

    else:
        if G.coords.shape[1] == 2:
            pg.plot(G.coords, pen=None, symbol='o')
        if G.coords.shape[1] == 3:
            pg.plot(G.coords[:, 0], G.coords[:, 1], G.coords[:, 2], 'bo')


def plot_pointcloud(P):
    r"""
    Plot the coordinates of a pointcloud.

    Parameters
    ----------
    P : PointsClouds object

    Examples
    --------
    >>> from pygsp import graphs, plotting
    >>> logo = graphs.PointsCloud('logo')
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


def plot_filter(filters, G=None, npoints=1000, line_width=4, x_width=3,
                x_size=10, plot_eigenvalues=None, show_sum=None,
                savefig=False, plot_name=None):
    r"""
    Plot a system of graph spectral filters.

    Parameters
    ----------
    filters : filter object
    G : Graph object
        If not specified it will take the one used to create the filter
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
    MexicanHat : has to compute lmax
    >>> try:
    ...     plotting.plot_filter(mh)
    ... except:
    ...     pass

    """
    if not isinstance(filters.g, list):
        filters.g = [filters.g]
    if plot_eigenvalues is None:
        plot_eigenvalues = hasattr(G, 'e')
    if show_sum is None:
        show_sum = len(filters.g) > 1
    if G is None:
        G = filters.G
    if plot_name is None:
        plot_name = "Filter plot of " + G.gtype

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
        plt.close(fig)
    # else:
    #     plt.show()


def plot_signal(G, signal, show_edges=None, cp={-6, -3, 160},
                vertex_size=None, vertex_highlight=False, climits=None,
                colorbar=True, bar=False, bar_width=1, savefig=False,
                plot_name=None):
    r"""
    Plot a graph signal in 2D or 3D.

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
        To plot an extra line showing the sum of the squared magnitudes\
         of the filters (default True if there is multiple filters).
    bar : int
        NOT IMPLEMENTED: 0 display color, 1 display bar for the graph
        (default 0).
    bar_width : int
        Width of the bar (default 1).
    savefig : boolean
        Determine wether the plot is saved as a PNG file in your\
         current directory (True) or shown in a window (False) (default False).
    plot_name : str
        To give custom names to plots

    Examples
    --------
    >>> import numpy as np
    >>> import pygsp
    >>> from pygsp import graphs
    >>> from pygsp import filters
    >>> from pygsp import plotting
    >>> G = graphs.Ring(15)
    >>> signal = np.sin((np.arange(1, 16)*2*np.pi/15))
    >>> try:
    ...     plotting.plot_signal(signal, G)
    ... except:
    ...     pass


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
        climits = {cmin, cmax}
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
                # ax.plot(x, y, color=G.plotting['edge_color'], marker='o', markerfacecolor=G.plotting['vertex_color'])
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
                    # ax.plot(x3, y3, z3, color=G.plotting['edge_color'], marker='o', markerfacecolor=G.plotting['vertex_color'])

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
        plt.close(fig)
    # else:
    #     plt.show()


def pg_plot_signal(G, signal, show_edges=None, cp={-6, -3, 160},
                   vertex_size=None, vertex_highlight=False, climits=None,
                   colorbar=True, bar=False, bar_width=1):
    r"""
    Plot a graph signal in 2D or 3D, with pyqtgraph.

    Parameters
    ----------
    G : Graph object
        If not specified it will take the one used to create the filter.
    signal : array of int
        Signal applied to the graph.
    show_edges : boolean
        Set to 0 to only draw the vertices (default G.Ne < 10000).
    cp : List of int
        Camera position for a 3D graph.
    vertex_size : int
        Size of circle representing each signal component.
    vertex_highlight : boolean
        Vector of indices of vertices to be highlighted.
    climits : array of int
        Limits of the colorbar.
    colorbar : boolean
        To plot an extra line showing the sum of the squared magnitudes\
         of the filters (default True if there is multiple filters).
    bar : int
        0 display color, 1 display bar for the graph (default 0).
    bar_width : int
        Width of the bar (default 1).

    Examples
    --------
    TODO

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
        climits = {cmin, cmax}

    # pygtgraph window initialization in 2D and 3D
    global window_list
    if 'window_list' not in globals():
        window_list = {}

    if G.coords.shape[1] == 2:
        w = pg.GraphicsWindow()
        w.setWindowTitle(G.gtype)
        v = w.addViewBox()
    elif G.coords.shape[1] == 3:
        app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.opts['distance'] = 10
        w.show()
        w.setWindowTitle(G.gtype)

    # Plot signal
    pos = np.array([0., 1., 0.5, 0.25, 0.75])
    color = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255],
                      (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, color)

    mininum = min(signal)
    maximum = max(signal)

    normalized_signal = map(lambda x: (float(x) - mininum) / (maximum - mininum), signal)

    if G.coords.shape[1] == 2:
        gp = pg.ScatterPlotItem(G.coords[:, 0], G.coords[:, 1], size=vertex_size, brush=cmap.map(normalized_signal, 'qcolor'))
        v.addItem(gp)
    if G.coords.shape[1] == 3:
        gp = gl.GLScatterPlotItem(G.coords[:, 0], G.coords[:, 1],
                                  G.coords[:, 2], size=vertex_size, c=signal)
        w.addItem(gp)

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
                app = QtGui.QApplication([])
                w = gl.GLViewWidget()
                w.opts['distance'] = 10
                w.show()
                w.setWindowTitle(G.gtype)

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

    # Multiple windows handling
    if G.coords.shape[1] == 2:
        window_list[str(uuid.uuid4())] = w
    elif G.coords.shape[1] == 3:
        window_list[str(uuid.uuid4())] = app


def rescale_center(x):
    r"""
    Rescaling the dataset.

    Rescaling the dataset, previously and mainly used in the SwissRoll graph.

    Parameters
    ----------
    x : ndarray
        Dataset to be rescaled.

    Returns
    -------
    r : ndarray
        Rescaled dataset.

    Examples
    --------
    >>> import pygsp
    >>> pygsp.utils.dummy(0, [1, 2, 3], True)
    array([1, 2, 3])

    """
    N = x.shape[1]
    # TODO virer d?
    d = x.shape[0]
    y = x - np.kron(np.ones((1, N)), np.expand_dims(np.mean(x, axis=1),
                                                    axis=1))
    c = np.amax(y)
    r = y / c

    return r
