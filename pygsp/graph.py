#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Graph:
	'''
	The Graph class is composed of two main components : a SpectralProp object and an AttributeMap object.
	'''
	def __init__(self, spectralProp, attributeMap):
		'''
		Constructor of Graph
		'''
		self.spectralProp = spectralProp
		self.attributeMap = attributeMap

class SpectralProp:
	'''
	The SpectralProp class is used to represent the mathematical formulations of the graph.
	The only mandatory argument is the Laplacian
	'''
	def __init__(self, laplacian, weightMatrix = None, eigVects = None, eigVals = None):
		'''
		Constructor of SpectralProp
		'''
		self.laplacian = laplacian
		self.weightMatrix = weightMatrix
		self.eigVects = eigVects
		self.eigVals = eigVals
		self.lambdaMax = None

	def computeEigDecomp(self, force = False):
		'''
		The function computes the eigen values and eigen vectors of the ``laplacian``. 
		If they already exists, the function simply returns. This can be bypassed by setting ``force`` to True.  
		TODO sort eigVals and eigVects
		'''
		if force or self.eigVals is None or self.eigVects is None:
			self.eigVals, self.eigVects = np.linalg.eigh(self.laplacian.todense())

	def getLargestEigVal(self, force = False):
		'''
		The function compute the largest eigen value of the ``laplacian``
		'''
		if force or self.lambdaMax is None:
			if self.eigVals is None:
				self.lambdaMax = np.linalg.eigvalsh(self.laplacian.todense()).max()
			else:
				self.lambdaMax = self.eigVals.max()
		return self.lambdaMax


class AttributeMap:
	'''
	The AttributeMap class is used to store the labels and coordinates map of the graph.
	'''
	def __init__(self):
		'''
		Constructor of AttributeMap
		'''
		self.attributes = dict(dict())

	def __init__(self, attributes):
		'''
		Constructor of AttributeMap
		'''
		self.attributes = attributes

	def readAttributes(self, path):
		'''
		This function is used to fill the dictionnary  ``attributes``
		TODO
		'''

def createGraphFromWeight(weightMatrix):
	'''
	This function uses a weight Matrix as input and creates a Graph object
	'''

def laplacian(weightMatrix, laplacianType = 'raw'):
	'''
	This function computes the laplacian of a graph from its weight matrix
	Mostly inspired by https://github.com/aweinstein/PySGWT/blob/master/sgwt.py
	'''
	N = weightMatrix.shape[0]
    # TODO: Raise exception if A is not square

    degrees = weightMatrix.sum(1)
    # To deal with loops, must extract diagonal part of A
    diagw = np.diag(weightMatrix)

    # w will consist of non-diagonal entries only
    ni2, nj2 = weightMatrix.nonzero()
    w2 = weightMatrix[ni2, nj2]
    ndind = (ni2 != nj2).nonzero() # Non-diagonal indices
    ni = ni2[ndind]
    nj = nj2[ndind]
    w = w2[ndind]

    di = np.arange(N) # diagonal indices

    if laplacian_type == 'raw':
        # non-normalized laplaciand L = D - A
        L = np.diag(degrees - diagw)
        L[ni, nj] = -w
        L = lil_matrix(L)
    elif laplacian_type == 'normalized':
        # TODO: Implement the normalized laplacian case
        # % normalized laplacian D^(-1/2)*(D-A)*D^(-1/2)
        # % diagonal entries
        # dL=(1-diagw./degrees); % will produce NaN for degrees==0 locations
        # dL(degrees==0)=0;% which will be fixed here
        # % nondiagonal entries
        # ndL=-w./vec( sqrt(degrees(ni).*degrees(nj)) );
        # L=sparse([ni;di],[nj;di],[ndL;dL],N,N);
        print 'Not implemented'
    else:
        # TODO: Raise an exception
        print "Don't know what to do"

    return L
