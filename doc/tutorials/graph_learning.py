import numpy as np
import pygsp
from matplotlib import pyplot as plt

#%%
n_signals = 4
def f1(x, y): return (-np.sin((2-x-y)**2)/2 + np.cos(y*3))/2
def f2(x, y): return np.cos((x+y)**2);
def f3(x, y): return ((x-.5)**2 + (y-.5)**3 + x - y)*3;
def f4(x, y): return np.sin(3*((x-.5)**2+(y-.5)**2));



#%%
n_nodes = 300
coords = np.random.uniform(size=(n_nodes, 2))

# the vectors of smooth signals will be stored as columns of X
X = np.zeros((n_nodes, n_signals))
for i, f in enumerate((f1, f2, f3, f4)):
    X[:, i] = f(coords[:, 0], coords[:, 1])



#%%
plt.figure(figsize=(16, 10))
for i, f in enumerate((f1, f2, f3, f4)):
    plt.subplot(2, 2, i+1);
    plt.scatter(coords[:, 0], coords[:, 1], 500, X[:, i], '.');
    plt.title('smooth signal {}'.format(i+1));
    plt.axis('off');
    plt.colorbar();


#%%

k = 5
param_opt = {'maxit':2000}

G1 = pygsp.graphs.LearnedFromSmoothSignals(X[:, 0], coords=coords, k=k, param_opt=param_opt)
G2 = pygsp.graphs.LearnedFromSmoothSignals(X[:, 1], coords=coords, k=k, param_opt=param_opt)
G3 = pygsp.graphs.LearnedFromSmoothSignals(X[:, 2], coords=coords, k=k, param_opt=param_opt)
G4 = pygsp.graphs.LearnedFromSmoothSignals(X[:, 3], coords=coords, k=k, param_opt=param_opt)

#%%


# G1.set_coordinates()
plt.figure(figsize=(18, 16))
for i, G in enumerate((G1, G2, G3, G4)):
    _, _, weights = G.get_edge_list()
    ax = plt.subplot(2, 2, i+1);
    G.plot(vertex_size=50, edge_width=weights, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.title('n_vertices = {}, n_edges = {}, k = {:.1f}'.format(G.n_vertices, G.n_edges, G.n_edges * 2 / G.n_vertices))



#%%

G_all = pygsp.graphs.LearnedFromSmoothSignals(X, coords=coords, k=k, param_opt=param_opt)
plt.figure(figsize=(8, 6))
ax = plt.axes()
G_all.plot(vertex_size=0, ax=ax)
plt.xticks([]), plt.yticks([])
plt.title('n_vertices = {}, n_edges = {}, k = {:.1f}'.format(G_all.n_vertices, G_all.n_edges, G_all.n_edges * 2 / G_all.n_vertices))



#%%

G_coords = pygsp.graphs.LearnedFromSmoothSignals(coords, coords=coords, k=k, param_opt=param_opt)
plt.figure(figsize=(8, 6))
ax = plt.axes()
G_coords.plot(vertex_size=0, ax=ax, edges=True)




#%%


import scipy
Z = scipy.spatial.distance_matrix(X, X)**2
plt.imshow(Z)
print(np.count_nonzero(Z))
edge_mask = (Z < 1 / np.sqrt(n_nodes)).astype(np.int)
np.fill_diagonal(edge_mask, 0)
print(np.count_nonzero(edge_mask))
edge_mask[:3,:3]



#%%

G_all = pygsp.graphs.LearnedFromSmoothSignals(X, coords=coords, k=k, 
                                              param_opt=param_opt, 
                                              sparse=True, edge_mask=edge_mask)
plt.figure(figsize=(8, 6))
ax = plt.axes()
G_all.plot(vertex_size=0, ax=ax)
plt.xticks([]), plt.yticks([])
plt.title('n_vertices = {}, n_edges = {}, k = {:.1f}'.format(G_all.n_vertices, G_all.n_edges, G_all.n_edges * 2 / G_all.n_vertices))




#%%






#%%






#%%






#%%






#%%


