==============
Simple problem
==============

This example demonstrates how to create a graph, a filter and analyse a signal on the graph.

>>> import pygsp
>>> G = pygsp.graphs.Logo()
>>> f = pygsp.filters.Heat(G)
>>> Sl = f.analysis(G, G.L.todense(), method='cheby')
