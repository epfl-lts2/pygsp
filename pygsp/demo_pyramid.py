import pygsp
import numpy as np
from pygsp import plotting, graphs, operators, utils, filters

G = graphs.Sensor(512, distribute=True)

filters = [lambda x: 5./(5 + x)]

Gs = operators.kron_pyramid(G, 5, epsilon=0.1)
operators.compute_fourier_basis(Gs)

f = np.ones((G.N))
f[np.arange(G.N/2)] = -1
f = f + 10*Gs[0].U[:, 7]

f2 = np.ones((G.N, 2))
f2[np.arange(G.N/2)] = -1

ca, pe = operators.pyramid_analysis(Gs, f, filters=filters, verbose=False)
ca2, pe2 = operators.pyramid_analysis(Gs, f2, filters=filters, verbose=False)

coeff = operators.pyramid_cell2coeff(ca, pe)
coeff2 = operators.pyramid_cell2coeff(ca2, pe2)

f_pred, _ = operators.pyramid_synthesis(Gs, coeff, verbose=False)
f_pred2, _ = operators.pyramid_synthesis(Gs, coeff2, verbose=False)

err = np.linalg.norm(f_pred-f)/np.linalg.norm(f)
err2 = np.linalg.norm(f_pred2-f2)/np.linalg.norm(f2)
print('erreur de f (1d) : {}'.format(err))
print('erreur de f2 (2d) : {}'.format(err2))
