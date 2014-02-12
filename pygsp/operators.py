#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def gft(g, signal):  
    """Graph Fourier transform

        Compute the full graph Fourier transform using the eigenvectors of the graph laplacian
        hatf = gft(g,f)
        
        Warning, the graph eigenvectors needs to be first computed. For a graph g, you can do this using

        g.compute_eig_decomp

        Parameters
        ----------
        g : graph
        signal : signal (numpy array)


        Returns
        -------
        hatf : graph Fourier tranform 
        """

    if g.U == None:
        raise "Error, you need to compute the eigenvectors of the graph first!"

    hatf = np.dot( g.U.conj().T , signal)

    return hatf


def igft(g, signal):  
    """Inverse graph Fourier transform

        Compute the full inverse graph Fourier transform using the eigenvectors of the graph laplacian
        f = igft(g,hatf)
        
        Warning, the graph eigenvectors needs to be first computed. For a graph g, you can do this using

        g.compute_eig_decomp

        Parameters
        ----------
        g : graph
        signal : signal (numpy array)

        Returns
        -------
        hatf : graph Fourier tranform 
        """

    if g.U == None:
        raise "Error, you need to compute the eigenvectors of the graph first!"

    hatf = np.dot( g.U , signal)
   

    return hatf