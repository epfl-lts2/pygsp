# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt


def plot_graph(G):
    # TODO handling when G is a list of graphs
    # TODO integrate param when G is a clustered graph

    # TODO Fix this condition
    if True:
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
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0), np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0), np.expand_dims(G.coords[kj, 1], axis=0)))
                plt.plot(x, y)
                plt.show()
            else:
                raise NotImplementedError('TODO')
