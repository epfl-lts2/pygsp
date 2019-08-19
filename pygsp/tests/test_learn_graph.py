import unittest
import numpy as np
from nn import nn, sparse_distance_matrix
from pygsp._nearest_neighbor import nearest_neighbor as nn
from pygsp._nearest_neighbor import sparse_distance_matrix
from scipy import sparse
import pygsp as pg
from pygsp.utils import distanz
from pygsp.graphs.learngraph import *

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
            
    def test_learn_graph_log(self):
        # Create a bunch of signals
        n=100
        d = 400
        G = pg.graphs.Sensor(N=n,k=6)
        G.compute_fourier_basis()
        # g = pg.filters.Heat(G, scale=5)
        g = pg.filters.Filter(G,lambda x:1/(1+5*x))
        S = np.random.randn(n,d)
        X = np.squeeze(g.filter(S))
        
        Z = distanz(X.transpose())
        k=6
        theta, theta_min, theta_max = gsp_compute_graph_learning_theta(Z, k)
        learned_W, _ = learn_graph_log_degree(theta*Z, verbosity=0)
        
        neighbors, distances = nn(X, k=3*k, kind='knn')  
        np.testing.assert_equal(distances[:,0],0)
        dmat = distances[:,1:]
        theta2, theta_min2, theta_max2 = gsp_compute_graph_learning_theta(dmat, k)
        np.testing.assert_allclose(theta, theta2)
        np.testing.assert_allclose(theta_min, theta_min2)
        np.testing.assert_allclose(theta_max, theta_max2)
        W = sparse_distance_matrix(neighbors, distances)
        learned_W2, _ = learn_graph_log_degree(W*theta2, edge_mask=W>0, verbosity=0)
        assert(np.sum(np.abs(learned_W2.todense()-learned_W))/np.sum(np.abs(learned_W))<1e-3)
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
if __name__ == '__main__':
    unittest.main()