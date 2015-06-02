==============
Simple problem
==============

This example demonstrates how to create a graph, a filter and analyse a signal on the graph.

>>> import pygsp
>>> G = pygsp.graphs.Logo()
>>> f = pygsp.filters.Heat(G)
<class 'pygsp.filters.Heat'> : has to compute lmax
>>> Sl = f.analysis(G, G.L.todense(), method='cheby')
