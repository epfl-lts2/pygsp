# -*- coding: utf-8 -*-

"""
Test suite for the Import and Export functionality inside the graphs module of the pygsp package.

"""

import unittest

import numpy as np
import networkx as nx
import graph_tool as gt
import random

from pygsp import graphs

class TestCase(unittest.TestCase):

    def test_networkx_export_import(self):
        #Export to networkx and reimport to PyGSP

        #Exporting the Bunny graph
        g = graphs.Bunny()
        g_nx = g.to_networkx()
        g2 = graphs.Graph.from_networkx(g_nx)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

    def test_networkx_import_export(self):
        #Import from networkx then export to networkx again
        g_nx = nx.gnm_random_graph(100, 50) #Generate a random graph
        g = graphs.Graph.from_networkx(g_nx).to_networkx()

        assert nx.is_isomorphic(g_nx, g)
        np.testing.assert_array_equal(nx.adjacency_matrix(g_nx).todense(),
                                      nx.adjacency_matrix(g).todense())

    def test_graphtool_export_import(self):
        #Export to graph tool and reimport to PyGSP directly
        #The exported graph is a simple one without an associated Signal
        g = graphs.Bunny()
        g_gt = g.to_graphtool()
        g2 = graphs.Graph.from_graphtool(g_gt)
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())


    def test_graphtool_multiedge_import(self):
        #Manualy create a graph with multiple edges
        g_gt = gt.Graph()
        g_gt.add_vertex(10)
        #connect edge (3,6) three times
        for i in range(3):
            g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
        g = graphs.Graph.from_graphtool(g_gt)
        assert g.W[3,6] == 3.0

        #test custom aggregator function
        g2 = graphs.Graph.from_graphtool(g_gt, aggr_fun=np.mean)
        assert g2.W[3,6] == 1.0

        eprop_double = g_gt.new_edge_property("double")

        #Set the weight of 2 out of the 3 edges. The last one has a default weight of 0
        e = g_gt.edge(3,6, all_edges=True)
        eprop_double[e[0]] = 8.0
        eprop_double[e[1]] = 1.0

        g_gt.edge_properties["weight"] = eprop_double
        g3 = graphs.Graph.from_graphtool(g_gt, aggr_fun=np.mean)
        assert g3.W[3,6] == 3.0

    def test_graphtool_import_export(self):
        # Import to PyGSP and export again to graph tool directly
        # create a random graphTool graph that does not contain multiple edges and no signal
        g_gt = gt.Graph()
        g_gt.add_vertex(100)

        # insert single random links
        eprop_double = g_gt.new_edge_property("double")
        for s, t in set(zip(np.random.randint(0, 100, 100),
                        np.random.randint(0, 100, 100))):
            g_gt.add_edge(g_gt.vertex(s), g_gt.vertex(t))

        for e in g_gt.edges():
            eprop_double[e] = random.random()
        g_gt.edge_properties["weight"] = eprop_double

        g2_gt = graphs.Graph.from_graphtool(g_gt).to_graphtool()

        assert len([e for e in g_gt.edges()]) == len([e for e in g2_gt.edges()]), \
            "the number of edge does not correspond"

        key = lambda e: str(e.source()) + ":" + str(e.target())
        for e1, e2 in zip(sorted(g_gt.edges(), key=key), sorted(g2_gt.edges(), key=key)):
            assert e1.source() == e2.source()
            assert e1.target() == e2.target()
        for v1, v2 in zip(g_gt.vertices(), g2_gt.vertices()):
            assert v1 == v2

    def test_networkx_singal_export(self):
        logo = graphs.Logo()
        s = np.random.random(logo.N)
        s2 = np.random.random(logo.N)
        logo.set_signal(s, "signal1")
        logo.set_signal(s2, "signal2")
        logo_nx = logo.to_networkx()
        for i in range(50):
            # Randomly check the signal of 50 nodes to see if they are the same
            rd_node = np.random.randint(logo.N)
            assert logo_nx.node[rd_node]["signal1"] == s[rd_node]
            assert logo_nx.node[rd_node]["signal2"] == s2[rd_node]

    def test_graphtool_signal_export(self):
        g = graphs.Logo()
        s = np.random.random(g.N)
        s2 = np.random.random(g.N)
        g.set_signal(s, "signal1")
        g.set_signal(s2, "signal2")
        g_gt = g.to_graphtool()
        #Check the signals on all nodes
        for i, v in enumerate(g_gt.vertices()):
            assert g_gt.vertex_properties["signal1"][v] == s[i]
            assert g_gt.vertex_properties["signal2"][v] == s2[i]
    def test_graphtool_signal_import(self):
        g_gt = gt.Graph()
        g_gt.add_vertex(10)

        g_gt.add_edge(g_gt.vertex(3), g_gt.vertex(6))
        g_gt.add_edge(g_gt.vertex(4), g_gt.vertex(6))
        g_gt.add_edge(g_gt.vertex(7), g_gt.vertex(2))

        vprop_double = g_gt.new_vertex_property("double")

        vprop_double[g_gt.vertex(0)] = 5
        vprop_double[g_gt.vertex(1)] = -3
        vprop_double[g_gt.vertex(2)] = 2.4

        g_gt.vertex_properties["signal"] = vprop_double
        g = graphs.Graph.from_graphtool(g_gt, singals_names=["signal"])
        assert g.signals["signal"][0] == 5.0
        assert g.signals["signal"][1] == -3.0
        assert g.signals["signal"][2] == 2.4

    def test_networkx_singal_import(self):
        g_nx = nx.Graph()
        g_nx.add_edge(3,4)
        g_nx.add_edge(2,4)
        g_nx.add_edge(3,5)
        print(list(g_nx.node)[0])
        dic_signal = {
            2 : 4.0,
            3 : 5.0,
            4 : 3.3,
            5 : 2.3
        }

        nx.set_node_attributes(g_nx, dic_signal, "signal1")
        g = graphs.Graph.from_networkx(g_nx, singals_names=["signal1"])

        nodes_mapping = list(g_nx.node)
        for i in range(len(nodes_mapping)):
            assert g.signals["signal1"][i] == nx.get_node_attributes(g_nx, "signal1")[nodes_mapping[i]]


    def test_save_load(self):
        g = graphs.Bunny()
        g.save("bunny.gml")
        g2 = graphs.Graph.load("bunny.gml")
        np.testing.assert_array_equal(g.W.todense(), g2.W.todense())

suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)