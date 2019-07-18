import unittest
import numpy as np
from nn import nn, sparse_distance_matrix
from learn_graph import  prox_sum_log, isvector, issymetric, squareform_sp, sum_squareform, lin_map, norm_S
from scipy import sparse

class TestCase(unittest.TestCase):
    def test_prox_sum_log(self):
        np.testing.assert_array_less(0,prox_sum_log(np.random.randn((10)),0.1))
    def test_isvector(self):
        assert(isvector(4)==False)
        assert(isvector([4,4])==False)
        assert(isvector(np.array([4,4,4]))==True)
        assert(isvector(np.zeros([3,4]))==False)
        assert(isvector(np.zeros([3]))==True)
        assert(isvector(np.zeros([3,0]))==False)
        assert(isvector(np.zeros([3,1]))==False)
        assert(isvector(sparse.csr_matrix((10,1)))==True)
        
    def test_issymetric(self):
        data = np.random.RandomState(42).uniform(size=(100, 3))
        # neighbors, distances = nn(data, backend='flann')
        neighbors, distances = nn(data)
        W = sparse_distance_matrix(neighbors, distances)
        assert(issymetric(W))
        W3 = W.copy()
        W3[:,0] = -W3[:,0]
        assert(issymetric(W3)==False)
        
    def test_squareform_sp(self):
        data = np.random.RandomState(42).uniform(size=(100, 3))
        # neighbors, distances = nn(data, backend='flann')
        neighbors, distances = nn(data)
        W = sparse_distance_matrix(neighbors, distances)
        w = squareform_sp(W)
        W2 = squareform_sp(w)
        np.testing.assert_array_almost_equal(W.todense(), W2.todense())
        
    def test_sum_squareform(self):
        nt =10
        # a) Create a random symetric matrix
        A = np.random.rand(nt,nt)
        A = A +A.transpose()
        A = A - np.diag(np.diag(A))
        # b) Make the test
        res1 = np.sum(A, axis=1)
        a = squareform_sp(A)
        S,St = sum_squareform(nt)
        res2 = S @ a
        np.testing.assert_allclose(res1,res2)
        np.testing.assert_array_equal(S.transpose().todense(), St.todense())
        
    def test_sum_squareform_sparse(self):
        data = np.random.uniform(size=(100, 3))
        # neighbors, distances = nn(data, backend='flann')
        neighbors, distances = nn(data)
        W = sparse_distance_matrix(neighbors, distances)
        nt = W.shape[0]
        # b) Make the test
        res1 = np.squeeze(np.asarray(np.sum(W.todense(), axis=1)))
        w = squareform_sp(W)
        S,St = sum_squareform(nt, w)
        wfull = w.data
        res2 = S @ wfull
        np.testing.assert_allclose(res1,res2)
        np.testing.assert_array_equal(S.transpose().todense(), St.todense())
        
    def test_linmap(self):
        np.testing.assert_array_almost_equal(lin_map(np.arange(10)/10,[0,10],[0,1]),np.arange(10))
        np.testing.assert_array_almost_equal(lin_map(np.arange(11)/10,[0,10]),np.arange(11))
        
    def test_norm_S(self):
        edge_mask = None
        for n in range(15):
            S, St = sum_squareform(10, mask=edge_mask)
            res1 = np.linalg.norm(S.todense(),2)
            res2 = norm_S(S)
            np.testing.assert_allclose(res1,res2)

suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
if __name__ == '__main__':
    unittest.main()