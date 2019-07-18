import unittest
import numpy as np
from nn import nn

class TestCase(unittest.TestCase):
    def test_nngraph(self, n_vertices=24):
        data = np.random.RandomState(42).uniform(size=(n_vertices, 3))
        metrics = ['euclidean', 'manhattan', 'max_dist', 'minkowski']
        backends = ['scipy-kdtree', 'scipy-ckdtree', 'flann', 'nmslib']

        for metric in metrics:
            for kind in ['knn', 'radius']:
                for backend in backends:
                    params = dict(features=data, metric=metric, kind=kind, radius=0.25)
                    ref_nn, ref_d = nn(backend='scipy-pdist', **params)
                    # Unsupported combinations.
                    if backend == 'flann' and metric == 'max_dist':
                        self.assertRaises(ValueError, nn, data,
                                          metric=metric, backend=backend)
                    elif backend == 'nmslib' and metric == 'minkowski':
                        self.assertRaises(ValueError, nn, data,
                                          metric=metric, backend=backend)
                    elif backend == 'nmslib' and kind == 'radius':
                        self.assertRaises(ValueError, nn, data,
                                          kind=kind, backend=backend)
                    else:
                        params['backend'] = backend
                        if backend == 'flann':
#                             params['target_precision'] = 1
                            other_nn, other_d = nn(random_seed=44, **params)
                        else:
                            other_nn, other_d = nn(**params)
                        print(kind, backend)
                        for a,b in zip(ref_nn, other_nn):
                            np.testing.assert_allclose(np.sort(a),np.sort(b), rtol=1e-5)

                        for a,b in zip(ref_d, other_d):
                            np.testing.assert_allclose(np.sort(a),np.sort(b), rtol=1e-5)
                            
    def test_sparse_distance_matrix(self):
        data = np.random.RandomState(42).uniform(size=(24, 3))
                            
                            
suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
if __name__ == '__main__':
    unittest.main()