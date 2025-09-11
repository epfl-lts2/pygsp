import os

import numpy as np


def _import_networkx():
    try:
        import networkx as nx
    except Exception as e:
        raise ImportError(
            "Cannot import networkx. Use graph-tool or try to "
            "install it with pip (or conda) install networkx. "
            "Original exception: {}".format(e)
        )
    return nx


def _import_graphtool():
    try:
        import graph_tool as gt
    except Exception as e:
        raise ImportError(
            "Cannot import graph-tool. Use networkx or try to "
            "install it. Original exception: {}".format(e)
        )
    return gt


class IOMixIn:
    def _break_signals(self):
        r"""Break N-dimensional signals into N 1D signals."""
        for name in list(self.signals.keys()):
            if self.signals[name].ndim == 2:
                for i, signal_1d in enumerate(self.signals[name].T):
                    self.signals[name + "_" + str(i)] = signal_1d
                del self.signals[name]

    def _join_signals(self):
        r"""Join N 1D signals into one N-dimensional signal."""
        joined = dict()
        for name in self.signals:
            name_base = name.rsplit("_", 1)[0]
            names = joined.get(name_base, list())
            names.append(name)
            joined[name_base] = names
        for name_base, names in joined.items():
            if len(names) > 1:
                names = sorted(names)  # ensure dim ordering (_0, _1, etc.)
                signal_nd = np.stack([self.signals[n] for n in names], axis=1)
                self.signals[name_base] = signal_nd
                for name in names:
                    del self.signals[name]

    def to_networkx(self):
        r"""Export the graph to NetworkX.

        Edge weights are stored as an edge attribute,
        under the name "weight".

        Signals are stored as node attributes,
        under their name in the :attr:`signals` dictionary.
        `N`-dimensional signals are broken into `N` 1-dimensional signals.
        They will eventually be joined back together on import.

        Returns
        -------
        graph : :class:`networkx.Graph`
            A NetworkX graph object.

        See Also
        --------
        to_graphtool : export to graph-tool
        save : save to a file

        Examples
        --------
        >>> import networkx as nx
        >>> from matplotlib import pyplot as plt
        >>> graph = graphs.Path(4, directed=True)
        >>> graph.set_signal(np.full(4, 2.3), 'signal')
        >>> graph = graph.to_networkx()
        >>> print(nx.info(graph))
        DiGraph named 'Path' with 4 nodes and 3 edges
        >>> nx.is_directed(graph)
        True
        >>> graph.nodes()
        NodeView((0, 1, 2, 3))
        >>> graph.edges()
        OutEdgeView([(0, 1), (1, 2), (2, 3)])
        >>> graph.nodes()[2]
        {'signal': 2.3}
        >>> graph.edges()[(0, 1)]
        {'weight': 1.0}
        >>> # nx.draw(graph, with_labels=True)

        Another common goal is to use NetworkX to compute some properties to be
        be imported back in the PyGSP as signals.

        >>> import networkx as nx
        >>> from matplotlib import pyplot as plt
        >>> graph = graphs.Sensor(100, seed=42)
        >>> graph.set_signal(graph.coords, 'coords')
        >>> graph = graph.to_networkx()
        >>> betweenness = nx.betweenness_centrality(graph, weight='weight')
        >>> nx.set_node_attributes(graph, betweenness, 'betweenness')
        >>> graph = graphs.Graph.from_networkx(graph)
        >>> graph.compute_fourier_basis()
        >>> graph.set_coordinates(graph.signals['coords'])
        >>> fig, axes = plt.subplots(1, 2)
        >>> _ = graph.plot(graph.signals['betweenness'], ax=axes[0])
        >>> _ = axes[1].plot(graph.e, graph.gft(graph.signals['betweenness']))

        """
        nx = _import_networkx()

        def convert(number):
            # NetworkX accepts arbitrary python objects as attributes, but:
            # * the GEXF writer does not accept any NumPy types (on signals),
            # * the GraphML writer does not accept NumPy ints.
            if issubclass(number.dtype.type, (np.integer, np.bool_)):
                return int(number)
            else:
                return float(number)

        def edges():
            for source, target, weight in zip(*self.get_edge_list()):
                yield int(source), int(target), {"weight": convert(weight)}

        def nodes():
            for vertex in range(self.n_vertices):
                signals = {
                    name: convert(signal[vertex])
                    for name, signal in self.signals.items()
                }
                yield vertex, signals

        self._break_signals()
        graph = nx.DiGraph() if self.is_directed() else nx.Graph()
        graph.add_nodes_from(nodes())
        graph.add_edges_from(edges())
        graph.name = self.__class__.__name__
        return graph

    def to_graphtool(self):
        r"""Export the graph to graph-tool.

        Edge weights are stored as an edge property map,
        under the name "weight".

        Signals are stored as vertex property maps,
        under their name in the :attr:`signals` dictionary.
        `N`-dimensional signals are broken into `N` 1-dimensional signals.
        They will eventually be joined back together on import.

        Returns
        -------
        graph : :class:`graph_tool.Graph`
            A graph-tool graph object.

        See Also
        --------
        to_networkx : export to NetworkX
        save : save to a file

        Examples
        --------
        >>> import graph_tool as gt
        >>> import graph_tool.draw
        >>> from matplotlib import pyplot as plt
        >>> graph = graphs.Path(4, directed=True)
        >>> graph.set_signal(np.full(4, 2.3), 'signal')
        >>> graph = graph.to_graphtool()
        >>> graph.is_directed()
        True
        >>> graph.vertex_properties['signal'][2]
        2.3
        >>> graph.edge_properties['weight'][graph.edge(0, 1)]
        1.0
        >>> # gt.draw.graph_draw(graph, vertex_text=graph.vertex_index)

        Another common goal is to use graph-tool to compute some properties to
        be imported back in the PyGSP as signals.

        >>> import graph_tool as gt
        >>> import graph_tool.centrality
        >>> from matplotlib import pyplot as plt
        >>> graph = graphs.Sensor(100, seed=42)
        >>> graph.set_signal(graph.coords, 'coords')
        >>> graph = graph.to_graphtool()
        >>> vprop, eprop = gt.centrality.betweenness(
        ...     graph, weight=graph.edge_properties['weight'])
        >>> graph.vertex_properties['betweenness'] = vprop
        >>> graph = graphs.Graph.from_graphtool(graph)
        >>> graph.compute_fourier_basis()
        >>> graph.set_coordinates(graph.signals['coords'])
        >>> fig, axes = plt.subplots(1, 2)
        >>> _ = graph.plot(graph.signals['betweenness'], ax=axes[0])
        >>> _ = axes[1].plot(graph.e, graph.gft(graph.signals['betweenness']))

        """

        gt = _import_graphtool()
        graph = gt.Graph(directed=self.is_directed())

        sources, targets, weights = self.get_edge_list()
        graph.add_edge_list(zip(sources, targets))
        prop = graph.new_edge_property(gt._gt_type(weights.dtype))
        prop.get_array()[:] = weights
        graph.edge_properties["weight"] = prop

        self._break_signals()
        for name, signal in self.signals.items():
            prop = graph.new_vertex_property(gt._gt_type(signal.dtype))
            prop.get_array()[:] = signal
            graph.vertex_properties[name] = prop

        return graph

    @classmethod
    def from_networkx(cls, graph, weight="weight"):
        r"""Import a graph from NetworkX.

        Edge weights are retrieved as an edge attribute,
        under the name specified by the ``weight`` parameter.

        Signals are retrieved from node attributes,
        and stored in the :attr:`signals` dictionary under the attribute name.
        `N`-dimensional signals that were broken during export are joined.

        Parameters
        ----------
        graph : :class:`networkx.Graph`
            A NetworkX graph object.
        weight : string or None, optional
            The edge attribute that holds the numerical values used as the edge
            weights. All edge weights are set to 1 if None, or not found.

        Returns
        -------
        graph : :class:`~pygsp.graphs.Graph`
            A PyGSP graph object.

        Notes
        -----

        The nodes are ordered according to :meth:`networkx.Graph.nodes`.

        In NetworkX, node attributes need not be set for every node.
        If a node attribute is not set for a node, a NaN is assigned to the
        corresponding signal for that node.

        If the graph is a :class:`networkx.MultiGraph`, multiedges are
        aggregated by summation.

        See Also
        --------
        from_graphtool : import from graph-tool
        load : load from a file

        Examples
        --------
        >>> import networkx as nx
        >>> graph = nx.Graph()
        >>> graph.add_edge(1, 2, weight=0.2)
        >>> graph.add_edge(2, 3, weight=0.9)
        >>> graph.add_node(4, sig=3.1416)
        >>> graph.nodes()
        NodeView((1, 2, 3, 4))
        >>> graph = graphs.Graph.from_networkx(graph)
        >>> graph.W.toarray()
        array([[0. , 0.2, 0. , 0. ],
               [0.2, 0. , 0.9, 0. ],
               [0. , 0.9, 0. , 0. ],
               [0. , 0. , 0. , 0. ]])
        >>> graph.signals
        {'sig': array([   nan,    nan,    nan, 3.1416])}

        """
        nx = _import_networkx()
        from .graph import Graph

        adjacency = nx.to_scipy_sparse_array(graph, weight=weight)
        graph_pg = Graph(adjacency)

        for i, node in enumerate(graph.nodes()):
            for name in graph.nodes[node].keys():
                try:
                    signal = graph_pg.signals[name]
                except KeyError:
                    signal = np.full(graph_pg.n_vertices, np.nan)
                    graph_pg.set_signal(signal, name)
                try:
                    signal[i] = graph.nodes[node][name]
                except KeyError:
                    pass  # attribute not set for node

        graph_pg._join_signals()
        return graph_pg

    @classmethod
    def from_graphtool(cls, graph, weight="weight"):
        r"""Import a graph from graph-tool.

        Edge weights are retrieved as an edge property,
        under the name specified by the ``weight`` parameter.

        Signals are retrieved from node properties,
        and stored in the :attr:`signals` dictionary under the property name.
        `N`-dimensional signals that were broken during export are joined.

        Parameters
        ----------
        graph : :class:`graph_tool.Graph`
            A graph-tool graph object.
        weight : string
            The edge property that holds the numerical values used as the edge
            weights. All edge weights are set to 1 if None, or not found.

        Returns
        -------
        graph : :class:`~pygsp.graphs.Graph`
            A PyGSP graph object.

        Notes
        -----

        If the graph has multiple edge connecting the same two nodes, a sum
        over the edges is taken to merge them.

        See Also
        --------
        from_networkx : import from NetworkX
        load : load from a file

        Examples
        --------
        >>> import graph_tool as gt
        >>> graph = gt.Graph(directed=False)
        >>> e1 = graph.add_edge(0, 1)
        >>> e2 = graph.add_edge(1, 2)
        >>> v = graph.add_vertex()
        >>> eprop = graph.new_edge_property("double")
        >>> eprop[e1] = 0.2
        >>> eprop[graph.edge(1, 2)] = 0.9
        >>> graph.edge_properties["weight"] = eprop
        >>> vprop = graph.new_vertex_property("double", val=np.nan)
        >>> vprop[3] = 3.1416
        >>> graph.vertex_properties["sig"] = vprop
        >>> graph = graphs.Graph.from_graphtool(graph)
        >>> graph.W.toarray()
        array([[0. , 0.2, 0. , 0. ],
               [0.2, 0. , 0.9, 0. ],
               [0. , 0.9, 0. , 0. ],
               [0. , 0. , 0. , 0. ]])
        >>> graph.signals
        {'sig': PropertyArray([   nan,    nan,    nan, 3.1416])}

        """
        gt = _import_graphtool()
        import graph_tool.spectral

        from .graph import Graph

        weight = graph.edge_properties.get(weight, None)
        adjacency = gt.spectral.adjacency(graph, weight=weight)
        graph_pg = Graph(adjacency.T)

        for name, signal in graph.vertex_properties.items():
            graph_pg.set_signal(signal.get_array(), name)

        graph_pg._join_signals()
        return graph_pg

    @classmethod
    def load(cls, path, fmt=None, backend=None):
        r"""Load a graph from a file.

        Edge weights are retrieved as an edge attribute named "weight".

        Signals are retrieved from node attributes,
        and stored in the :attr:`signals` dictionary under the attribute name.
        `N`-dimensional signals that were broken during export are joined.

        Parameters
        ----------
        path : string
            Path to the file from which to load the graph.
        fmt : {'graphml', 'gml', 'gexf', None}, optional
            Format in which the graph is saved.
            Guessed from the filename extension if None.
        backend : {'networkx', 'graph-tool', None}, optional
            Library used to load the graph. Automatically chosen if None.

        Returns
        -------
        graph : :class:`Graph`
            The loaded graph.

        See Also
        --------
        save : save a graph to a file
        from_networkx : load with NetworkX then import in the PyGSP
        from_graphtool : load with graph-tool then import in the PyGSP

        Notes
        -----

        A lossless round-trip is only guaranteed if the graph (and its signals)
        is saved and loaded with the same backend.

        Loading from other formats is possible by loading in NetworkX or
        graph-tool, and importing to the PyGSP.
        The proposed formats are however tested for faithful round-trips.

        Examples
        --------
        >>> graph = graphs.Logo()
        >>> graph.save('logo.graphml')
        >>> graph = graphs.Graph.load('logo.graphml')
        >>> import os
        >>> os.remove('logo.graphml')

        """

        if fmt is None:
            fmt = os.path.splitext(path)[1][1:]
        if fmt not in ["graphml", "gml", "gexf"]:
            raise ValueError(f"Unsupported format {fmt}.")

        def load_networkx(path, fmt):
            nx = _import_networkx()
            load = getattr(nx, "read_" + fmt)
            graph = load(path)
            return cls.from_networkx(graph)

        def load_graphtool(path, fmt):
            gt = _import_graphtool()
            graph = gt.load_graph(path, fmt=fmt)
            return cls.from_graphtool(graph)

        if backend == "networkx":
            return load_networkx(path, fmt)
        elif backend == "graph-tool":
            return load_graphtool(path, fmt)
        elif backend is None:
            try:
                return load_networkx(path, fmt)
            except ImportError:
                try:
                    return load_graphtool(path, fmt)
                except ImportError:
                    raise ImportError("Cannot import networkx nor graph-tool.")
        else:
            raise ValueError(f"Unknown backend {backend}.")

    def save(self, path, fmt=None, backend=None):
        r"""Save the graph to a file.

        Edge weights are stored as an edge attribute,
        under the name "weight".

        Signals are stored as node attributes,
        under their name in the :attr:`signals` dictionary.
        `N`-dimensional signals are broken into `N` 1-dimensional signals.
        They will eventually be joined back together on import.

        Supported formats are:

        * GraphML_, a comprehensive XML format.
          Supported by NetworkX_, graph-tool_, NetworKit_, igraph_, Gephi_,
          Cytoscape_, SocNetV_.
        * GML_ (Graph Modelling Language), a simple non-XML format.
          Supported by NetworkX_, graph-tool_, NetworKit_, igraph_, Gephi_,
          Cytoscape_, SocNetV_, Tulip_.
        * GEXF_ (Graph Exchange XML Format), Gephi's XML format.
          Supported by NetworkX_, NetworKit_, Gephi_, Tulip_, ngraph_.

        If unsure, we recommend GraphML_.

        .. _GraphML: https://en.wikipedia.org/wiki/GraphML
        .. _GML: https://en.wikipedia.org/wiki/Graph_Modelling_Language
        .. _GEXF: https://gexf.net/
        .. _NetworkX: https://networkx.org
        .. _graph-tool: https://graph-tool.skewed.de
        .. _NetworKit: https://networkit.github.io
        .. _igraph: https://igraph.org
        .. _ngraph: https://github.com/anvaka/ngraph
        .. _Gephi: https://gephi.org
        .. _Cytoscape: https://cytoscape.org
        .. _SocNetV: https://socnetv.org
        .. _Tulip: https://tulip.labri.fr

        Parameters
        ----------
        path : string
            Path to the file where the graph is to be saved.
        fmt : {'graphml', 'gml', 'gexf', None}, optional
            Format in which to save the graph.
            Guessed from the filename extension if None.
        backend : {'networkx', 'graph-tool', None}, optional
            Library used to load the graph. Automatically chosen if None.

        See Also
        --------
        load : load a graph from a file
        to_networkx : export as a NetworkX graph, and save with NetworkX
        to_graphtool : export as a graph-tool graph, and save with graph-tool

        Notes
        -----

        A lossless round-trip is only guaranteed if the graph (and its signals)
        is saved and loaded with the same backend.

        Saving in other formats is possible by exporting to NetworkX or
        graph-tool, and using their respective saving functionality.
        The proposed formats are however tested for faithful round-trips.

        Edge weights and signal values are rounded at the sixth decimal when
        saving in ``fmt='gml'`` with ``backend='graph-tool'``.

        Examples
        --------
        >>> graph = graphs.Logo()
        >>> graph.save('logo.graphml')
        >>> graph = graphs.Graph.load('logo.graphml')
        >>> import os
        >>> os.remove('logo.graphml')

        """

        if fmt is None:
            fmt = os.path.splitext(path)[1][1:]
        if fmt not in ["graphml", "gml", "gexf"]:
            raise ValueError(f"Unsupported format {fmt}.")

        def save_networkx(graph, path, fmt):
            nx = _import_networkx()
            graph = graph.to_networkx()
            save = getattr(nx, "write_" + fmt)
            save(graph, path)

        def save_graphtool(graph, path, fmt):
            graph = graph.to_graphtool()
            graph.save(path, fmt=fmt)

        if backend == "networkx":
            save_networkx(self, path, fmt)
        elif backend == "graph-tool":
            save_graphtool(self, path, fmt)
        elif backend is None:
            try:
                save_networkx(self, path, fmt)
            except ImportError:
                try:
                    save_graphtool(self, path, fmt)
                except ImportError:
                    raise ImportError("Cannot import networkx nor graph-tool.")
        else:
            raise ValueError(f"Unknown backend {backend}.")
