# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            if G.coords.shape[1] == 3:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                x = np.concatenate((np.expand_dims(G.coords[ki, 0], axis=0), np.expand_dims(G.coords[kj, 0], axis=0)))
                y = np.concatenate((np.expand_dims(G.coords[ki, 1], axis=0), np.expand_dims(G.coords[kj, 1], axis=0)))
                z = np.concatenate((np.expand_dims(G.coords[ki, 2], axis=0), np.expand_dims(G.coords[kj, 2], axis=0)))
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
                    ax.plot(x3, y3, z3)
                ax.plot(x2, y2, z2, 'ro')
                plt.show()
