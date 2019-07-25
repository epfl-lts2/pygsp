import unittest
import numpy as np
from pygsp._nearest_neighbor import nearest_neighbor, sparse_distance_matrix

class TestCase(unittest.TestCase):
    def test_nngraph(self, n_vertices=100):
        data = np.random.RandomState(42).uniform(size=(n_vertices, 3))
        metrics = ['euclidean', 'manhattan', 'max_dist', 'minkowski']
        backends = ['scipy-kdtree', 'scipy-ckdtree', 'flann', 'nmslib']

        for metric in metrics:
            for kind in ['knn', 'radius']:
                for backend in backends:
                    params = dict(features=data, metric=metric, kind=kind, radius=0.25, k=10)
                    
                    ref_nn, ref_d = nearest_neighbor(backend='scipy-pdist', **params)
                    # Unsupported combinations.
                    if backend == 'flann' and metric == 'max_dist':
                        self.assertRaises(ValueError, nearest_neighbor, data,
                                          metric=metric, backend=backend)
                    elif backend == 'nmslib' and metric == 'minkowski':
                        self.assertRaises(ValueError, nearest_neighbor, data,
                                          metric=metric, backend=backend)
                    elif backend == 'nmslib' and kind == 'radius':
                        self.assertRaises(ValueError, nearest_neighbor, data,
                                          kind=kind, backend=backend)
                    else:
                        params['backend'] = backend
                        if backend == 'flann':
                            other_nn, other_d = nearest_neighbor(random_seed=44, target_precision=1, **params)
                        else:
                            other_nn, other_d = nearest_neighbor(**params)
                        print(kind, backend)
                        if backend == 'flann':
                            for a,b in zip(ref_nn, other_nn):
                                assert(len(set(a)-set(b))<=2)

                            for a,b in zip(ref_d, other_d):
                                np.testing.assert_allclose(np.mean(a).astype(np.float32),np.mean(b), atol=2e-2)                            
                        else:
                            for a,b,c,d in zip(ref_nn, other_nn, ref_d, other_d):
                                e = set(a)-set(b)
                                assert(len(e)<=1)
                                if len(e)==0:
                                    np.testing.assert_allclose(np.sort(c),np.sort(d), rtol=1e-5)

    def test_sparse_distance_matrix(self):
        data = np.random.RandomState(42).uniform(size=(200, 3))
        neighbors, distances = nearest_neighbor(data)
        W = sparse_distance_matrix(neighbors, distances, symmetrize=True)
        # Assert symetry
        np.testing.assert_allclose(W.todense(), W.T.todense())
        # positivity
        np.testing.assert_array_equal(W.todense()>=0, True)
        # 0 diag
        np.testing.assert_array_equal(np.diag(W.todense())==0, True)

        # Assert that it is not symmetric anymore
        W = sparse_distance_matrix(neighbors, distances, symmetrize=False)
        assert(np.sum(np.abs(W.todense()-W.T.todense()))>0.1)
        # positivity
        np.testing.assert_array_equal(W.todense()>=0, True)
        # 0 diag
        np.testing.assert_array_equal(np.diag(W.todense())==0, True)
        # everything is used once
        np.testing.assert_allclose(np.sum(W.todense()), np.sum(distances))

        # simple test with a kernel
        W = sparse_distance_matrix(neighbors, 1/(1+distances), symmetrize=True)
        # Assert symetry
        np.testing.assert_allclose(W.todense(), W.T.todense())
        # positivity
        np.testing.assert_array_equal(W.todense()>=0, True)
        # 0 diag
        np.testing.assert_array_equal(np.diag(W.todense())==0, True)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
