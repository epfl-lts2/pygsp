import pygsp
import numpy as np
from pygsp import graphs, reduction, data_handling

G = graphs.Sensor(512, distribute=True)

g = [lambda x: 5./(5 + x)]

Gs = reduction.kron_pyramid(G, 5, epsilon=0.1)
graphs.gutils.compute_fourier_basis(Gs)

f = np.ones((G.N))
f[np.arange(G.N/2)] = -1
f = f + 10*Gs[0].U[:, 7]

f2 = np.ones((G.N, 2))
f2[np.arange(G.N/2)] = -1

ca, pe = reduction.pyramid_analysis(Gs, f, filters=g, verbose=False)
ca2, pe2 = reduction.pyramid_analysis(Gs, f2, filters=g, verbose=False)

coeff = data_handling.pyramid_cell2coeff(ca, pe)
coeff2 = data_handling.pyramid_cell2coeff(ca2, pe2)

f_pred, _ = reduction.pyramid_synthesis(Gs, coeff, verbose=False)
f_pred2, _ = reduction.pyramid_synthesis(Gs, coeff2, verbose=False)

err = np.linalg.norm(f_pred-f)/np.linalg.norm(f)
err2 = np.linalg.norm(f_pred2-f2)/np.linalg.norm(f2)
print('erreur de f (1d) : {}'.format(err))
print('erreur de f2 (2d) : {}'.format(err2))
